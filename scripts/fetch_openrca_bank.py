#!/usr/bin/env python3
# pyright: basic
"""Fetch a frozen OpenRCA Bank suite from the official ZIP or a pinned mirror."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import struct
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

try:
    from scripts.prepare_bank_suite import mirror_source_index, plan_cases
except ModuleNotFoundError:  # Direct `python scripts/fetch_openrca_bank.py` use.
    from prepare_bank_suite import mirror_source_index, plan_cases

TAIL_BYTES = 65_557
SPEC_PATH = Path(__file__).resolve().parents[1] / "benchmarks/openrca-bank-d9-v1.json"
SOURCE_FILES = (
    ("metric/metric_app.csv", 1000, "tc"),
    ("metric/metric_container.csv", 1000, "cmdb_id"),
    ("log/log_service.csv", 1000, "cmdb_id"),
    ("trace/trace_span.csv", 1, "cmdb_id"),
)
URL_TEMPLATE = (
    "https://drive.usercontent.google.com/download?"
    "id={file_id}&export=download&confirm=t"
)
RangeFetcher = Callable[[str, int, int, int, Path], None]
MirrorFetcher = Callable[[str, Path, int, str], tuple[str, str]]
TZ = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class ZipEntry:
    name: str
    size: int
    compressed_size: int
    offset: int
    crc32: str
    method: int


class TailView(io.RawIOBase):
    """Seekable view that exposes an archive tail at its original offsets."""

    def __init__(self, tail: bytes, total: int) -> None:
        self.tail = tail
        self.total = total
        self.base = total - len(tail)
        self.position = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.total + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if self.position < self.base:
            raise ValueError("central directory extends beyond the downloaded tail")
        stop = self.total if size < 0 else min(self.total, self.position + size)
        result = self.tail[self.position - self.base : stop - self.base]
        self.position = stop
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_integrity(path: Path) -> tuple[int, str, str]:
    digest = hashlib.sha256()
    crc = 0
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return size, digest.hexdigest(), f"{crc & 0xFFFFFFFF:08x}"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_content_range(value: str | None, start: int, end: int, total: int) -> None:
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", value or "", re.I)
    actual = tuple(map(int, match.groups())) if match else None
    if actual != (start, end, total):
        raise RuntimeError(
            f"unexpected Content-Range {value!r}; expected bytes {start}-{end}/{total}"
        )


def download_http_range(
    url: str, start: int, end: int, total: int, destination: Path
) -> None:
    """Download one exact inclusive range and reject whole-file/shifted responses."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    expected = end - start + 1
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=240) as response:  # noqa: S310
            validate_content_range(
                response.headers.get("Content-Range"), start, end, total
            )
            count = 0
            with temporary.open("wb") as stream:
                while chunk := response.read(min(1024 * 1024, expected - count + 1)):
                    count += len(chunk)
                    if count > expected:
                        raise RuntimeError("HTTP range response exceeded expected size")
                    stream.write(chunk)
        if count != expected:
            raise RuntimeError(
                f"HTTP range response size {count} does not match expected {expected}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_mirror_file(
    url: str, part: Path, expected_size: int, expected_sha256: str
) -> tuple[str, str]:
    """Resume one pinned mirror file and return its verified SHA-256 and CRC32."""
    if part.is_symlink():
        raise ValueError("mirror partial file must not be a symlink")
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected_size:
        raise ValueError("mirror partial file is larger than the pinned source")
    if offset < expected_size:
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={offset}-",
                "User-Agent": "RootWalk/0.1 pinned OpenRCA fetch",
            },
        )
        with urllib.request.urlopen(request, timeout=240) as response:  # noqa: S310
            status = response.getcode()
            if offset and status != 206:
                raise RuntimeError("mirror did not honor the resume Range request")
            if status == 206:
                value = response.headers.get("Content-Range")
                match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", value or "")
                actual = tuple(map(int, match.groups())) if match else None
                if actual != (offset, expected_size - 1, expected_size):
                    raise RuntimeError(f"unexpected mirror Content-Range: {value!r}")
            elif status != 200 or offset != 0:
                raise RuntimeError(f"unexpected mirror HTTP status: {status}")
            mode = "ab" if offset else "wb"
            with part.open(mode) as stream:
                while chunk := response.read(1024 * 1024):
                    if stream.tell() + len(chunk) > expected_size:
                        raise RuntimeError("mirror response exceeded pinned file size")
                    stream.write(chunk)
    size, digest, crc32 = _file_integrity(part)
    if size != expected_size or digest != expected_sha256:
        if size == expected_size:
            part.unlink(missing_ok=True)
        raise ValueError("mirror file size or SHA-256 does not match the pinned spec")
    return digest, crc32


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {name}")
    if path.name.lower() == "record.csv":
        raise ValueError("record.csv is evaluation-only and must never be downloaded")


def required_members(spec: dict[str, Any]) -> list[str]:
    """Build the allowlist: query plus four telemetry files per window date."""
    raw_cases = spec.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("suite spec cases must be a list")
    dates: set[str] = set()
    for case in raw_cases:
        if not isinstance(case, dict):
            raise ValueError("suite spec case must be an object")
        allowed = case.get("allowed_query_range")
        if not isinstance(allowed, dict) or allowed.get("end_exclusive") is not True:
            raise ValueError("suite case requires a half-open allowed_query_range")
        try:
            start = datetime.fromisoformat(str(allowed["start"])).astimezone(TZ)
            end = datetime.fromisoformat(str(allowed["end"])).astimezone(TZ)
        except (KeyError, ValueError) as exc:
            raise ValueError("suite case allowed_query_range is invalid") from exc
        if end <= start:
            raise ValueError("suite case allowed_query_range is invalid")
        current = start.date()
        last = (end - timedelta(microseconds=1)).date()
        while current <= last:
            dates.add(current.strftime("%Y_%m_%d"))
            current += timedelta(days=1)
    members = ["Bank/query.csv"]
    members.extend(
        f"Bank/telemetry/{source_date}/{relative}"
        for source_date in sorted(dates)
        for relative, _multiplier, _entity in SOURCE_FILES
    )
    for member in members:
        _safe_member(member)
    return members


def _inside_git_worktree(path: Path) -> bool:
    return any(
        (parent / ".git").is_file() or (parent / ".git/HEAD").is_file()
        for parent in (path, *path.parents)
    )


def _validate_destination(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError("source-root must not be a symlink")
    resolved = root.expanduser().resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError("refusing to write data at a filesystem root")
    if _inside_git_worktree(resolved):
        raise ValueError("source-root must be outside every Git working tree")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _validate_archive(path: Path, expected_bytes: int) -> Path:
    if path.is_symlink():
        raise ValueError("archive must not be a symlink")
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("archive must be a regular file")
    if _inside_git_worktree(resolved):
        raise ValueError("archive must be outside every Git working tree")
    if resolved.stat().st_size != expected_bytes:
        raise ValueError("archive size does not match the frozen suite")
    return resolved


def _copy_local_range(
    archive: Path, start: int, end: int, total: int, destination: Path
) -> None:
    if archive.stat().st_size != total:
        raise ValueError("archive size changed during extraction")
    expected = end - start + 1
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with archive.open("rb") as source, temporary.open("wb") as target:
            source.seek(start)
            remaining = expected
            while remaining:
                block = source.read(min(remaining, 1024 * 1024))
                if not block:
                    raise EOFError("local archive range is truncated")
                target.write(block)
                remaining -= len(block)
        if temporary.stat().st_size != expected:
            raise RuntimeError("local archive range has the wrong size")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_directory(root: Path, relative: str) -> Path:
    candidate = root
    for part in Path(relative).parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"data directory must not be a symlink: {relative}")
        candidate.mkdir(exist_ok=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"data directory escapes source-root: {relative}") from exc
    resolved.chmod(0o700)
    return resolved


def _load_index(
    archive: dict[str, Any],
    cache: Path,
    fetch_range: RangeFetcher,
    source_location: str | None = None,
) -> tuple[list[ZipEntry], int, str]:
    try:
        file_id = str(archive["file_id"])
        total = int(archive["bytes"])
        expected_directory_hash = str(archive["central_directory_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("suite archive metadata is incomplete") from exc
    if total <= 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_directory_hash):
        raise ValueError("suite archive metadata is invalid")
    url = source_location or URL_TEMPLATE.format(file_id=file_id)
    tail_size = min(TAIL_BYTES, total)
    start = total - tail_size
    tail_path = cache / "bank-tail.bin"
    if tail_path.is_symlink():
        raise ValueError("cached ZIP tail must not be a symlink")
    if not tail_path.exists() or tail_path.stat().st_size != tail_size:
        fetch_range(url, start, total - 1, total, tail_path)
    tail = tail_path.read_bytes()
    if len(tail) != tail_size:
        raise RuntimeError("cached ZIP tail has the wrong size")
    end_offset = tail.rfind(b"PK\x05\x06")
    if end_offset < 0:
        raise RuntimeError("ZIP end-of-central-directory record was not found")
    eocd = struct.unpack_from("<4s4H2IH", tail, end_offset)
    directory_size = eocd[5]
    directory_offset = eocd[6]
    relative = directory_offset - start
    if relative < 0 or relative + directory_size > len(tail):
        raise RuntimeError("ZIP central directory is not contained in the fixed tail")
    directory = tail[relative : relative + directory_size]
    actual_hash = hashlib.sha256(directory).hexdigest()
    if actual_hash != expected_directory_hash:
        raise RuntimeError("archive central directory SHA-256 changed")
    with zipfile.ZipFile(TailView(tail, total)) as archive_file:
        entries = [
            ZipEntry(
                name=info.filename,
                size=info.file_size,
                compressed_size=info.compress_size,
                offset=info.header_offset,
                crc32=f"{info.CRC:08x}",
                method=info.compress_type,
            )
            for info in archive_file.infolist()
        ]
    _write_json(cache / "bank-zip-index.json", [asdict(entry) for entry in entries])
    return entries, directory_offset, url


def _destination(root: Path, name: str) -> Path:
    _safe_member(name)
    relative = PurePosixPath(name)
    if name == "Bank/query.csv":
        target = root / "private_source" / Path(*relative.parts)
    else:
        target = root / Path(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = target.parent.resolve(strict=True)
    resolved_parent.relative_to(root)
    if target.is_symlink():
        raise ValueError(f"destination must not be a symlink: {name}")
    return target


def _extract_member(compressed: Path, output: Path, entry: ZipEntry) -> tuple[int, str]:
    temporary = output.with_suffix(output.suffix + ".part")
    digest = hashlib.sha256()
    crc = 0
    count = 0
    try:
        with compressed.open("rb") as source:
            header = source.read(30)
            if len(header) != 30:
                raise RuntimeError(f"truncated ZIP local header for {entry.name}")
            fields = struct.unpack("<4s5H3I2H", header)
            if fields[0] != b"PK\x03\x04" or fields[3] != entry.method:
                raise RuntimeError(f"unexpected ZIP local header for {entry.name}")
            if entry.method != zipfile.ZIP_DEFLATED:
                raise RuntimeError(
                    f"unsupported ZIP compression method for {entry.name}"
                )
            filename = source.read(fields[-2]).decode("utf-8")
            if filename != entry.name:
                raise RuntimeError(f"ZIP local filename mismatch for {entry.name}")
            source.read(fields[-1])
            decoder = zlib.decompressobj(-15)
            remaining = entry.compressed_size
            with temporary.open("wb") as target:
                while remaining:
                    block = source.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise EOFError(entry.name)
                    remaining -= len(block)
                    data = decoder.decompress(block)
                    crc = zlib.crc32(data, crc)
                    count += len(data)
                    digest.update(data)
                    target.write(data)
                data = decoder.flush()
                crc = zlib.crc32(data, crc)
                count += len(data)
                digest.update(data)
                target.write(data)
        if not decoder.eof or count != entry.size:
            raise RuntimeError(f"uncompressed size mismatch for {entry.name}")
        if f"{crc & 0xFFFFFFFF:08x}" != entry.crc32:
            raise RuntimeError(f"CRC mismatch for {entry.name}")
        temporary.replace(output)
        return count, digest.hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def _existing_result(
    output: Path, provenance: Path, entry: ZipEntry
) -> dict[str, Any] | None:
    if not output.is_file() or output.is_symlink() or not provenance.is_file():
        return None
    value = json.loads(provenance.read_text(encoding="utf-8"))
    if (
        value.get("name") == entry.name
        and value.get("size") == entry.size
        and value.get("crc32") == entry.crc32
        and value.get("sha256") == _sha256(output)
    ):
        return value
    return None


def _fetch_mirror_files(
    root: Path,
    spec: dict[str, Any],
    members: list[str],
    download: MirrorFetcher,
) -> list[dict[str, Any]]:
    mirror = spec.get("source_mirror")
    if not isinstance(mirror, dict):
        raise ValueError("--mirror requires a suite with source_mirror metadata")
    sources = mirror_source_index(spec)
    if set(sources) != set(members):
        raise ValueError("mirror source manifest does not match the frozen allowlist")
    template = str(mirror["url_template"])
    repository = str(mirror["repository"])
    commit = str(mirror["commit"])
    cache = _safe_directory(root, ".openrca/cache/mirror")
    provenance_root = _safe_directory(root, ".openrca/provenance")
    _safe_directory(root, "private_source")
    results: list[dict[str, Any]] = []
    for name in members:
        entry = sources[name]
        raw_size = entry.get("bytes")
        raw_sha256 = entry.get("sha256")
        if (
            isinstance(raw_size, bool)
            or not isinstance(raw_size, int)
            or not isinstance(raw_sha256, str)
        ):
            raise ValueError(f"invalid pinned mirror metadata for {name}")
        expected_size = raw_size
        expected_sha256 = raw_sha256
        url = template.format(
            repository=repository,
            commit=commit,
            path=name,
        )
        output = _destination(root, name)
        if output.exists():
            if output.is_symlink() or not output.is_file():
                raise ValueError(f"mirror destination must be a regular file: {name}")
            size, digest, crc32 = _file_integrity(output)
            if size != expected_size or digest != expected_sha256:
                raise ValueError(f"existing mirror file does not match spec: {name}")
        else:
            key = hashlib.sha256(name.encode()).hexdigest()[:16]
            part = cache / f"{key}.part"
            digest, crc32 = download(
                url,
                part,
                expected_size,
                expected_sha256,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part, output)
            size = expected_size
        if name == "Bank/query.csv":
            output.chmod(0o600)
        result = {
            "name": name,
            "size": size,
            "compressed_size": None,
            "offset": None,
            "crc32": crc32,
            "sha256": digest,
            "archive_bytes": None,
            "source_kind": "huggingface_mirror",
            "source_commit": commit,
            "source_url": url,
            "local_path": output.relative_to(root).as_posix(),
            "crc_computed": True,
            "sha256_verified": True,
            "verified_size": size,
        }
        key = hashlib.sha256(name.encode()).hexdigest()[:16]
        provenance = provenance_root / f"{key}.json"
        if provenance.is_symlink():
            raise ValueError(f"provenance file must not be a symlink: {name}")
        _write_json(provenance, result)
        results.append(result)
    return results


def fetch_openrca_bank(
    source_root: Path,
    spec_path: Path = SPEC_PATH,
    fetch_range: RangeFetcher = download_http_range,
    *,
    archive_path: Path | None = None,
    mirror: bool = False,
    mirror_fetcher: MirrorFetcher = download_mirror_file,
) -> list[dict[str, Any]]:
    """Fetch and verify the exact member allowlist required by the suite spec."""
    plan_cases(spec_path)
    root = _validate_destination(source_root)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    members = required_members(spec)
    if len(members) != len(set(members)):
        raise ValueError("frozen Bank suite contains duplicate source members")
    if mirror:
        if archive_path is not None:
            raise ValueError("--mirror and --archive are mutually exclusive")
        return _fetch_mirror_files(root, spec, members, mirror_fetcher)
    archive = spec.get("official_archive")
    if not isinstance(archive, dict):
        raise ValueError("suite spec has no official_archive object")
    cache = _safe_directory(root, ".openrca/cache")
    provenance_root = _safe_directory(root, ".openrca/provenance")
    _safe_directory(root, "private_source")
    if archive_path is not None:
        local_archive = _validate_archive(archive_path, int(archive["bytes"]))

        def local_range(
            _location: str,
            start: int,
            end: int,
            total: int,
            destination: Path,
        ) -> None:
            _copy_local_range(local_archive, start, end, total, destination)

        range_reader = local_range
        source_location = f"local-archive:{local_archive.name}"
    else:
        range_reader = fetch_range
        source_location = None
    entries, directory_offset, url = _load_index(
        archive, cache, range_reader, source_location
    )
    by_name = {entry.name: entry for entry in entries}
    offsets = sorted([entry.offset for entry in entries] + [directory_offset])
    results: list[dict[str, Any]] = []
    for name in members:
        entry = by_name.get(name)
        if entry is None:
            raise ValueError(
                f"required member is absent from the fixed archive: {name}"
            )
        key = hashlib.sha256(name.encode()).hexdigest()[:16]
        provenance = provenance_root / f"{key}.json"
        if provenance.is_symlink():
            raise ValueError(f"provenance file must not be a symlink: {name}")
        output = _destination(root, name)
        existing = _existing_result(output, provenance, entry)
        if existing is not None:
            if name == "Bank/query.csv":
                output.chmod(0o600)
            results.append(existing)
            continue
        next_offset = min(offset for offset in offsets if offset > entry.offset)
        end = next_offset - 1
        compressed = cache / f"{key}.member"
        if compressed.is_symlink():
            raise ValueError(f"cached member must not be a symlink: {name}")
        expected_range_bytes = end - entry.offset + 1
        if not compressed.exists() or compressed.stat().st_size != expected_range_bytes:
            range_reader(url, entry.offset, end, int(archive["bytes"]), compressed)
        size, sha256 = _extract_member(compressed, output, entry)
        if name == "Bank/query.csv":
            output.chmod(0o600)
        result = {
            **asdict(entry),
            "archive_bytes": int(archive["bytes"]),
            "central_directory_sha256": archive["central_directory_sha256"],
            "crc_verified": True,
            "http_range": [entry.offset, end],
            "local_path": output.relative_to(root).as_posix(),
            "sha256": sha256,
            "sha256_verified": _sha256(output) == sha256,
            "source_url": url,
            "verified_size": size,
        }
        if not result["sha256_verified"]:
            raise RuntimeError(f"SHA-256 verification failed for {name}")
        _write_json(provenance, result)
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--suite-spec", type=Path, default=SPEC_PATH)
    parser.add_argument(
        "--archive",
        type=Path,
        help="verified local Bank.zip; when set, no network request is made",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="download or verify the pinned Hugging Face mirror files",
    )
    args = parser.parse_args(argv)
    results = fetch_openrca_bank(
        args.source_root,
        args.suite_spec,
        archive_path=args.archive,
        mirror=args.mirror,
    )
    print(
        json.dumps(
            {
                "downloaded_or_verified": len(results),
                "members": [result["name"] for result in results],
                "source_root": str(args.source_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

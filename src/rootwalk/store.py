"""Minimal append-only run persistence and deterministic RunBundle export."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast


def _json_value(value: object) -> object:
    """Accept plain JSON values and Pydantic models without depending on Pydantic."""
    if hasattr(value, "model_dump"):
        return cast(Any, value).model_dump(mode="json")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_file(path: Path, value: object) -> None:
    path.write_text(f"{_canonical_json(value)}\n", encoding="utf-8")


def _validate_hash(content_hash: str) -> str:
    if len(content_hash) != 64 or any(
        character not in "0123456789abcdef" for character in content_hash
    ):
        raise ValueError("content_hash must be a lowercase SHA256 hex digest")
    return content_hash


def _merged_artifact_visibility(visibilities: Iterable[str]) -> str:
    observed = set(visibilities)
    allowed = {"model_readable", "audit_only", "evaluation_only"}
    if not observed <= allowed:
        raise ValueError(
            "artifact visibility must be model_readable, audit_only, or evaluation_only"
        )
    if "evaluation_only" in observed:
        return "evaluation_only"
    if observed == {"audit_only"}:
        return "audit_only"
    return "model_readable"


class StableKeyConflict(ValueError):
    """A stable key was reused for a different record."""


class RunSealedError(RuntimeError):
    """A sealed run rejected a domain mutation."""


class ArtifactBudgetExceeded(RuntimeError):
    """A new content-addressed artifact would exceed the run byte budget."""

    def __init__(self, *, limit: int, used: int, requested: int) -> None:
        self.limit = limit
        self.used = used
        self.requested = requested
        super().__init__(
            "artifact budget exceeded: "
            f"used={used}, requested={requested}, limit={limit}"
        )


@dataclass(frozen=True, slots=True)
class StoredRecord:
    revision: int
    kind: str
    stable_key: str
    payload: object


@dataclass(frozen=True, slots=True)
class SealState:
    run_id: str
    status: str
    report_ref: str
    report_revision: int
    sealed_revision: int


@dataclass(frozen=True, slots=True)
class CheckpointCursor:
    event_cursor: int
    projection_revision: int
    ledger_cursor: int
    phase: str
    pending_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutboxJob:
    run_id: str
    report_revision: int
    report_ref: str
    status: Literal["pending", "done"]


CheckpointRelation = Literal["domain_ahead", "consistent", "checkpoint_ahead"]


def checkpoint_relation(
    domain_revision: int, checkpoint_revision: int
) -> CheckpointRelation:
    """Classify recovery position without trusting either side."""
    if domain_revision < 0 or checkpoint_revision < 0:
        raise ValueError("revisions must be non-negative")
    if domain_revision > checkpoint_revision:
        return "domain_ahead"
    if domain_revision < checkpoint_revision:
        return "checkpoint_ahead"
    return "consistent"


class RunStore:
    """One SQLite append log for one run."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                revision INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                stable_key TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                report_ref TEXT NOT NULL,
                report_revision INTEGER NOT NULL,
                sealed_revision INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoint_cursor (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                event_cursor INTEGER NOT NULL,
                projection_revision INTEGER NOT NULL,
                ledger_cursor INTEGER NOT NULL,
                phase TEXT NOT NULL,
                pending_refs_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outbox (
                run_id TEXT NOT NULL,
                report_revision INTEGER NOT NULL,
                report_ref TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'done')),
                PRIMARY KEY (run_id, report_revision)
            );
            """
        )
        self._connection.commit()

    def append(self, kind: str, stable_key: str, payload: object) -> int:
        """Append once, or return the prior revision for an identical retry."""
        return self.append_many([(kind, stable_key, payload)])[0]

    def append_many(self, records: Iterable[tuple[str, str, object]]) -> list[int]:
        """Atomically append an idempotent batch in input order."""
        prepared = self._prepare_records(records)
        if not prepared:
            return []
        with self._connection:
            return self._append_prepared(prepared)

    @staticmethod
    def _prepare_records(
        records: Iterable[tuple[str, str, object]],
    ) -> list[tuple[str, str, str]]:
        prepared: list[tuple[str, str, str]] = []
        for kind, stable_key, payload in records:
            if not kind.strip() or not stable_key.strip():
                raise ValueError("kind and stable_key must be non-empty")
            prepared.append((kind, stable_key, _canonical_json(payload)))
        return prepared

    def _append_prepared(
        self,
        prepared: list[tuple[str, str, str]],
        *,
        allow_sealed_retry: bool = False,
    ) -> list[int]:
        revisions: list[int] = []
        sealed = self._connection.execute("SELECT 1 FROM runs LIMIT 1").fetchone()
        if sealed and any(kind != "late_audit" for kind, _, _ in prepared):
            if not allow_sealed_retry:
                raise RunSealedError("sealed runs only accept late_audit records")
            for kind, stable_key, payload_json in prepared:
                if kind == "late_audit":
                    continue
                row = self._connection.execute(
                    "SELECT kind, payload_json FROM records WHERE stable_key = ?",
                    (stable_key,),
                ).fetchone()
                if row is None:
                    raise RunSealedError("sealed runs only accept late_audit records")
                if row["kind"] != kind or row["payload_json"] != payload_json:
                    raise StableKeyConflict(
                        f"stable_key {stable_key!r} already identifies another record"
                    )
        for kind, stable_key, payload_json in prepared:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO records(kind, stable_key, payload_json)
                VALUES (?, ?, ?)
                """,
                (kind, stable_key, payload_json),
            )
            row = self._connection.execute(
                """
                SELECT revision, kind, payload_json
                FROM records
                WHERE stable_key = ?
                """,
                (stable_key,),
            ).fetchone()
            if row is None:
                raise RuntimeError("record insert did not produce a row")
            if row["kind"] != kind or row["payload_json"] != payload_json:
                raise StableKeyConflict(
                    f"stable_key {stable_key!r} already identifies another record"
                )
            revisions.append(cast(int, row["revision"]))
        return revisions

    @property
    def revision(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(revision), 0) AS revision FROM records"
        ).fetchone()
        if row is None:
            return 0
        return cast(int, row["revision"])

    @property
    def domain_revision(self) -> int:
        row = self._connection.execute(
            """
            SELECT COALESCE(MAX(revision), 0) AS revision
            FROM records
            WHERE kind != 'late_audit'
            """
        ).fetchone()
        if row is None:
            return 0
        return cast(int, row["revision"])

    def seal(self, run_id: str, status: str, report_ref: str) -> SealState:
        """Seal the run and enqueue its export in one transaction."""
        if not run_id.strip() or not status.strip() or not report_ref.strip():
            raise ValueError("run_id, status, and report_ref must be non-empty")
        with self._connection:
            return self._seal_prepared(run_id, status, report_ref)

    def append_many_and_seal(
        self,
        records: Iterable[tuple[str, str, object]],
        run_id: str,
        status: str,
        report_ref: str,
    ) -> tuple[list[int], SealState]:
        """Append records, seal the run, and enqueue export in one transaction."""
        if not run_id.strip() or not status.strip() or not report_ref.strip():
            raise ValueError("run_id, status, and report_ref must be non-empty")
        prepared = self._prepare_records(records)
        with self._connection:
            revisions = self._append_prepared(prepared, allow_sealed_retry=True)
            sealed = self._seal_prepared(run_id, status, report_ref)
        return revisions, sealed

    def _seal_prepared(self, run_id: str, status: str, report_ref: str) -> SealState:
        existing_row = self._connection.execute(
            """
            SELECT run_id, status, report_ref, report_revision, sealed_revision
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if existing_row is not None:
            existing = self._seal_state(existing_row)
            if existing.status != status or existing.report_ref != report_ref:
                raise RunSealedError("run is already sealed with another result")
            return existing
        other_run = self._connection.execute(
            "SELECT run_id FROM runs LIMIT 1"
        ).fetchone()
        if other_run is not None:
            raise RunSealedError("this single-run store is already sealed")
        report = self._connection.execute(
            """
            SELECT revision, kind
            FROM records
            WHERE stable_key = ?
            """,
            (report_ref,),
        ).fetchone()
        if report is None or report["kind"] != "report":
            raise ValueError("report_ref must identify a committed report record")
        report_revision = cast(int, report["revision"])
        sealed_revision = self.domain_revision
        self._connection.execute(
            """
            INSERT INTO runs(
                run_id, status, report_ref, report_revision, sealed_revision
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, status, report_ref, report_revision, sealed_revision),
        )
        self._connection.execute(
            """
            INSERT INTO outbox(
                run_id, report_revision, report_ref, status
            ) VALUES (?, ?, ?, 'pending')
            """,
            (run_id, report_revision, report_ref),
        )
        return SealState(run_id, status, report_ref, report_revision, sealed_revision)

    def seal_state(self, run_id: str) -> SealState | None:
        row = self._connection.execute(
            """
            SELECT run_id, status, report_ref, report_revision, sealed_revision
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        return self._seal_state(row) if row is not None else None

    def save_checkpoint(
        self,
        *,
        event_cursor: int,
        projection_revision: int,
        ledger_cursor: int,
        phase: str,
        pending_refs: Iterable[str],
    ) -> CheckpointCursor:
        cursors = (event_cursor, projection_revision, ledger_cursor)
        if any(type(cursor) is not int or cursor < 0 for cursor in cursors):
            raise ValueError("checkpoint cursors must be non-negative integers")
        if not phase.strip():
            raise ValueError("checkpoint phase must be non-empty")
        refs = tuple(pending_refs)
        if any(not ref.strip() for ref in refs):
            raise ValueError("pending_refs must contain non-empty strings")
        with self._connection:
            if any(cursor > self.domain_revision for cursor in cursors):
                raise ValueError("checkpoint cannot lead the current domain revision")
            self._connection.execute(
                """
                INSERT INTO checkpoint_cursor(
                    singleton, event_cursor, projection_revision, ledger_cursor,
                    phase, pending_refs_json
                ) VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    event_cursor = excluded.event_cursor,
                    projection_revision = excluded.projection_revision,
                    ledger_cursor = excluded.ledger_cursor,
                    phase = excluded.phase,
                    pending_refs_json = excluded.pending_refs_json
                """,
                (*cursors, phase, _canonical_json(list(refs))),
            )
        return CheckpointCursor(*cursors, phase, refs)

    def read_checkpoint(self) -> CheckpointCursor | None:
        row = self._connection.execute(
            """
            SELECT event_cursor, projection_revision, ledger_cursor, phase,
                   pending_refs_json
            FROM checkpoint_cursor
            WHERE singleton = 1
            """
        ).fetchone()
        if row is None:
            return None
        refs = cast(list[str], json.loads(cast(str, row["pending_refs_json"])))
        return CheckpointCursor(
            cast(int, row["event_cursor"]),
            cast(int, row["projection_revision"]),
            cast(int, row["ledger_cursor"]),
            cast(str, row["phase"]),
            tuple(refs),
        )

    def outbox_jobs(
        self, status: Literal["pending", "done"] | None = None
    ) -> list[OutboxJob]:
        if status is None:
            rows = self._connection.execute(
                """
                SELECT run_id, report_revision, report_ref, status
                FROM outbox
                ORDER BY run_id, report_revision
                """
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT run_id, report_revision, report_ref, status
                FROM outbox
                WHERE status = ?
                ORDER BY run_id, report_revision
                """,
                (status,),
            ).fetchall()
        return [self._outbox_job(row) for row in rows]

    def mark_outbox_done(self, run_id: str, report_revision: int) -> OutboxJob:
        with self._connection:
            updated = self._connection.execute(
                """
                UPDATE outbox
                SET status = 'done'
                WHERE run_id = ? AND report_revision = ?
                """,
                (run_id, report_revision),
            )
            if updated.rowcount != 1:
                raise KeyError((run_id, report_revision))
            row = self._connection.execute(
                """
                SELECT run_id, report_revision, report_ref, status
                FROM outbox
                WHERE run_id = ? AND report_revision = ?
                """,
                (run_id, report_revision),
            ).fetchone()
            if row is None:
                raise RuntimeError("outbox update did not produce a row")
        return self._outbox_job(row)

    def get(self, stable_key: str) -> StoredRecord | None:
        row = self._connection.execute(
            """
            SELECT revision, kind, stable_key, payload_json
            FROM records
            WHERE stable_key = ?
            """,
            (stable_key,),
        ).fetchone()
        return self._record(row) if row is not None else None

    def records(self, kind: str | None = None) -> list[StoredRecord]:
        if kind is None:
            rows = self._connection.execute(
                """
                SELECT revision, kind, stable_key, payload_json
                FROM records
                ORDER BY revision
                """
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT revision, kind, stable_key, payload_json
                FROM records
                WHERE kind = ?
                ORDER BY revision
                """,
                (kind,),
            ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: sqlite3.Row) -> StoredRecord:
        return StoredRecord(
            revision=cast(int, row["revision"]),
            kind=cast(str, row["kind"]),
            stable_key=cast(str, row["stable_key"]),
            payload=cast(object, json.loads(cast(str, row["payload_json"]))),
        )

    @staticmethod
    def _seal_state(row: sqlite3.Row) -> SealState:
        return SealState(
            cast(str, row["run_id"]),
            cast(str, row["status"]),
            cast(str, row["report_ref"]),
            cast(int, row["report_revision"]),
            cast(int, row["sealed_revision"]),
        )

    @staticmethod
    def _outbox_job(row: sqlite3.Row) -> OutboxJob:
        return OutboxJob(
            cast(str, row["run_id"]),
            cast(int, row["report_revision"]),
            cast(str, row["report_ref"]),
            cast(Literal["pending", "done"], row["status"]),
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


class ArtifactStore:
    """Content-addressed immutable files confined to one directory."""

    def __init__(self, root: Path, max_bytes: int = 64 * 1024 * 1024) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        requested = Path(root)
        if requested.is_symlink():
            raise ValueError("artifact root must not be a symlink")
        requested.mkdir(parents=True, exist_ok=True)
        if requested.is_symlink():
            raise ValueError("artifact root must not be a symlink")
        self.root = requested.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("artifact root must be a directory")
        root_stat = os.stat(self.root, follow_symlinks=False)
        self._root_identity = (root_stat.st_dev, root_stat.st_ino)
        self.max_bytes = max_bytes
        self._bytes_used = self._initialize_contents()

    @property
    def bytes_used(self) -> int:
        return self._bytes_used

    def set_max_bytes(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = max_bytes

    def put_bytes(self, content: bytes) -> str:
        content_hash = hashlib.sha256(content).hexdigest()
        root_fd = self._open_root_fd()
        temporary_name: str | None = None
        descriptor = -1
        try:
            try:
                existing_fd = os.open(
                    content_hash,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_fd,
                )
            except FileNotFoundError:
                existing_fd = None
            except OSError as exc:
                raise ValueError(
                    f"artifact is not a regular file: {content_hash}"
                ) from exc
            if existing_fd is not None:
                with os.fdopen(existing_fd, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ValueError(
                            f"artifact is not a regular file: {content_hash}"
                        )
                    if (
                        hashlib.file_digest(stream, "sha256").hexdigest()
                        != content_hash
                    ):
                        raise ValueError(f"artifact hash mismatch: {content_hash}")
                return content_hash
            if self._bytes_used + len(content) > self.max_bytes:
                raise ArtifactBudgetExceeded(
                    limit=self.max_bytes,
                    used=self._bytes_used,
                    requested=len(content),
                )
            while temporary_name is None:
                candidate = f".artifact-{secrets.token_hex(8)}"
                try:
                    descriptor = os.open(
                        candidate,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=root_fd,
                    )
                    temporary_name = candidate
                except FileExistsError:
                    continue
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary_name,
                content_hash,
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
            )
            self._assert_root_identity()
            self._bytes_used += len(content)
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=root_fd)
                except FileNotFoundError:
                    pass
            os.close(root_fd)
        return content_hash

    def put_text(self, content: str) -> str:
        return self.put_bytes(content.encode("utf-8"))

    def put_json(self, content: object) -> str:
        return self.put_text(_canonical_json(content))

    def path_for(self, content_hash: str) -> Path:
        self._assert_root_identity()
        path = self.root / _validate_hash(content_hash)
        if path.parent != self.root:
            raise ValueError("artifact path escapes its root")
        return path

    def committed_path(self, content_hash: str) -> Path:
        """Return a path verified now; production consumers should use read_bytes."""
        self.read_bytes(content_hash)
        return self.path_for(content_hash)

    def read_bytes(self, content_hash: str) -> bytes:
        name = _validate_hash(content_hash)
        root_fd = self._open_root_fd()
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise FileNotFoundError(
                        f"artifact is not committed: {content_hash}"
                    )
                content = stream.read()
        except OSError as exc:
            raise FileNotFoundError(
                f"artifact is not committed: {content_hash}"
            ) from exc
        finally:
            os.close(root_fd)
        if hashlib.sha256(content).hexdigest() != content_hash:
            raise ValueError(f"artifact hash mismatch: {content_hash}")
        return content

    def _existing_path(self, content_hash: str) -> Path:
        path = self.path_for(content_hash)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"artifact is not committed: {content_hash}")
        if path.resolve().parent != self.root:
            raise ValueError("artifact path escapes its root")
        return path

    def _assert_root_identity(self) -> None:
        if self.root.is_symlink():
            raise ValueError("artifact root changed after initialization")
        try:
            current = os.stat(self.root, follow_symlinks=False)
        except OSError as exc:
            raise ValueError("artifact root changed after initialization") from exc
        if (current.st_dev, current.st_ino) != self._root_identity:
            raise ValueError("artifact root changed after initialization")

    def _open_root_fd(self) -> int:
        flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(self.root, flags)
        except OSError as exc:
            raise ValueError("artifact root changed after initialization") from exc
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != self._root_identity:
            os.close(descriptor)
            raise ValueError("artifact root changed after initialization")
        return descriptor

    def _initialize_contents(self) -> int:
        root_fd = self._open_root_fd()
        total = 0
        try:
            for name in os.listdir(root_fd):
                item = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if name.startswith(".artifact-"):
                    if stat.S_ISREG(item.st_mode) or stat.S_ISLNK(item.st_mode):
                        os.unlink(name, dir_fd=root_fd)
                        continue
                    raise ValueError("artifact staging entry must not be a directory")
                if len(name) == 64 and all(
                    character in "0123456789abcdef" for character in name
                ):
                    if not stat.S_ISREG(item.st_mode):
                        raise ValueError("committed artifact must be a regular file")
                    total += item.st_size
        finally:
            os.close(root_fd)
        return total


class BundleExporter:
    """Export committed run records as a stable, static RunBundle."""

    _SINGULAR_FILES = {
        "task": "task.json",
        "topology": "topology.json",
        "report": "report.json",
    }
    _STREAM_FILES = {
        "decision": "decisions.jsonl",
        "attempt": "attempts.jsonl",
        "observation": "observations.jsonl",
        "evidence": "evidence.jsonl",
        "investigation_event": "investigation_events.jsonl",
        "context_manifest": "context_manifests.jsonl",
        "budget_ledger": "budget_ledger.jsonl",
        "model_request": "model_requests.jsonl",
        "late_audit": "execution_audit.jsonl",
    }
    _OPTIONAL_STREAM_FILES = {
        "evidence_assessment": "evidence_assessments.jsonl",
        "material_anomaly": "material_anomalies.jsonl",
        "anomaly_disposition": "anomaly_dispositions.jsonl",
        "reflection": "reflections.jsonl",
    }
    _ARTIFACT_REFERENCE_FIELDS = {
        "content_hash",
        "model_input_ref",
        "model_response_ref",
    }

    def __init__(self, store: RunStore, artifacts: ArtifactStore) -> None:
        self.store = store
        self.artifacts = artifacts

    def export(self, destination: Path) -> Path:
        singular_records: dict[str, StoredRecord] = {}
        for kind in self._SINGULAR_FILES:
            records = self.store.records(kind)
            if not records:
                raise ValueError(f"bundle requires a committed {kind} record")
            singular_records[kind] = records[-1]
        stream_files = dict(self._STREAM_FILES)
        stream_records = {kind: self.store.records(kind) for kind in self._STREAM_FILES}
        for kind, filename in self._OPTIONAL_STREAM_FILES.items():
            records = self.store.records(kind)
            if records:
                stream_files[kind] = filename
                stream_records[kind] = records
        manifest_records = self.store.records("manifest")
        exported_records = [*singular_records.values()]
        exported_records.extend(
            record for records in stream_records.values() for record in records
        )
        if manifest_records:
            exported_records.append(manifest_records[-1])
        artifact_index = self._artifact_index(exported_records)
        self._validate_artifact_references(exported_records, artifact_index)

        target = Path(destination).expanduser().resolve(strict=False)
        if destination.is_symlink():
            raise ValueError("bundle destination must not be a symlink")
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError("bundle destination must be empty")
        target.parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.tmp-"))
        try:
            artifact_root = root / "artifacts"
            artifact_root.mkdir()

            for kind, filename in self._SINGULAR_FILES.items():
                self._write_json(root / filename, singular_records[kind].payload)

            for kind, filename in stream_files.items():
                self._write_jsonl(root / filename, stream_records[kind])

            self._export_artifacts(artifact_root, artifact_index)
            self._write_json(root / "artifact_index.json", artifact_index)

            file_hashes = {
                path.relative_to(root).as_posix(): _sha256_file(path)
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }
            manifest: dict[str, object] = {}
            if manifest_records:
                payload = manifest_records[-1].payload
                if not isinstance(payload, dict):
                    raise ValueError("manifest payload must be a JSON object")
                manifest.update(cast(dict[str, object], payload))
            manifest.update(
                {
                    "artifact_index": "artifact_index.json",
                    "bundle_schema_version": "1",
                    "domain_cursor": self.store.domain_revision,
                    "file_hashes": file_hashes,
                }
            )
            self._write_json(root / "manifest.json", manifest)
            if target.exists():
                if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
                    raise FileExistsError("bundle destination must be empty")
                target.rmdir()
            os.replace(root, target)
            return target
        finally:
            if root.exists():
                shutil.rmtree(root)

    def _artifact_index(
        self, records: Iterable[StoredRecord]
    ) -> list[dict[str, object]]:
        visibilities: dict[str, set[str]] = {}
        for record in records:
            self._artifact_visibilities(record.payload, visibilities)
        index: list[dict[str, object]] = []
        for record in self.store.records("artifact"):
            raw_payload = record.payload
            if not isinstance(raw_payload, dict):
                raise ValueError("artifact payload must be a JSON object")
            payload = cast(dict[str, object], raw_payload)
            content_hash = payload.get("content_hash")
            storage = payload.get("storage")
            if not isinstance(content_hash, str):
                raise ValueError("artifact content_hash must be a string")
            _validate_hash(content_hash)
            if storage not in {"embedded", "external", "redacted"}:
                raise ValueError(
                    "artifact storage must be embedded, external, or redacted"
                )
            entry = dict(payload)
            entry["visibility"] = _merged_artifact_visibility(
                visibilities.get(content_hash, {"model_readable"})
            )
            entry.update({"revision": record.revision, "stable_key": record.stable_key})
            index.append(entry)
        return index

    def _export_artifacts(
        self, destination: Path, index: list[dict[str, object]]
    ) -> None:
        for entry in index:
            if entry["storage"] != "embedded":
                continue
            content_hash = cast(str, entry["content_hash"])
            target = destination / content_hash
            target.write_bytes(self.artifacts.read_bytes(content_hash))
            if _sha256_file(target) != content_hash:
                raise ValueError(f"artifact hash mismatch: {content_hash}")
            entry["bundle_path"] = f"artifacts/{content_hash}"

    def _validate_artifact_references(
        self,
        records: list[StoredRecord],
        artifact_index: list[dict[str, object]],
    ) -> None:
        registered = {entry["content_hash"] for entry in artifact_index}
        for record in records:
            for field, content_hash in self._artifact_references(record.payload):
                if not isinstance(content_hash, str) or content_hash not in registered:
                    raise ValueError(
                        f"unregistered artifact reference in {record.stable_key}: "
                        f"{field}={content_hash!r}"
                    )

    def _artifact_references(self, value: object) -> list[tuple[str, object]]:
        references: list[tuple[str, object]] = []
        if isinstance(value, dict):
            for key, nested in cast(dict[object, object], value).items():
                if key in self._ARTIFACT_REFERENCE_FIELDS and isinstance(key, str):
                    references.append((key, nested))
                references.extend(self._artifact_references(nested))
        elif isinstance(value, list):
            for nested in cast(list[object], value):
                references.extend(self._artifact_references(nested))
        return references

    def _artifact_visibilities(
        self, value: object, result: dict[str, set[str]]
    ) -> None:
        if isinstance(value, dict):
            item = cast(dict[object, object], value)
            content_hash = item.get("content_hash")
            visibility = item.get("visibility", "model_readable")
            if isinstance(content_hash, str) and isinstance(visibility, str):
                result.setdefault(content_hash, set()).add(visibility)
            for nested in item.values():
                self._artifact_visibilities(nested, result)
        elif isinstance(value, list):
            for nested in cast(list[object], value):
                self._artifact_visibilities(nested, result)

    @staticmethod
    def _record_payload(record: StoredRecord) -> dict[str, object]:
        return {
            "kind": record.kind,
            "payload": record.payload,
            "revision": record.revision,
            "stable_key": record.stable_key,
        }

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        _write_json_file(path, value)

    @classmethod
    def _write_jsonl(cls, path: Path, records: list[StoredRecord]) -> None:
        path.write_text(
            "".join(
                f"{_canonical_json(cls._record_payload(record))}\n"
                for record in records
            ),
            encoding="utf-8",
        )


def export_self_contained_bundle(
    bundle: Path, destination: Path, external_root: Path
) -> Path:
    """Copy a verified RunBundle and embed its non-evaluation external artifacts."""
    source = _trusted_directory(bundle, "bundle")
    external = _trusted_directory(external_root, "external-root")
    target = Path(destination).expanduser().resolve(strict=False)
    if destination.is_symlink() or target.exists():
        raise FileExistsError("self-contained bundle destination must not exist")
    if _paths_overlap(target, source) or _paths_overlap(target, external):
        raise ValueError(
            "self-contained bundle destination must not overlap its inputs"
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest = _json_object(source / "manifest.json", "bundle manifest")
    artifact_index_path, declared_paths = _verified_artifact_index(source, manifest)
    artifact_index = _json_list(artifact_index_path, "bundle artifact index")
    _reject_evaluation_only(artifact_index)
    artifact_paths = _artifact_body_paths(artifact_index)
    metadata_files = _verified_bundle_files(source, manifest, skip_paths=artifact_paths)
    domain_visibilities = _domain_artifact_visibilities(
        metadata_files, artifact_index_path
    )
    if any("evaluation_only" in values for values in domain_visibilities.values()):
        raise ValueError("self-contained export refuses evaluation_only artifacts")
    _validate_index_visibilities(artifact_index, domain_visibilities)
    _validate_embedded_layout(declared_paths, artifact_index)
    files = _verified_bundle_files(source, manifest)
    external_artifacts = _external_artifacts(external)

    stage = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.work-"))
    try:
        expected_hashes = cast(dict[str, str], manifest["file_hashes"])
        for relative, path in files.items():
            if relative in {"artifact_index.json", "manifest.json"}:
                continue
            copied = stage / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copied)
            if _sha256_file(copied) != expected_hashes[relative]:
                raise ValueError(f"copied bundle file hash mismatch: {relative}")

        artifact_root = stage / "artifacts"
        artifact_root.mkdir(exist_ok=True)
        embedded_external = 0
        for entry in artifact_index:
            storage = entry.get("storage")
            content_hash = entry.get("content_hash")
            if not isinstance(content_hash, str):
                raise ValueError("artifact index content_hash must be a string")
            _validate_hash(content_hash)
            if storage not in {"embedded", "external", "redacted"}:
                raise ValueError("artifact index has an invalid storage value")
            if storage != "external":
                continue
            source_artifact, size = _verified_external_artifact(
                external, external_artifacts, content_hash
            )
            copied = artifact_root / content_hash
            shutil.copyfile(source_artifact, copied)
            if copied.stat().st_size != size or _sha256_file(copied) != content_hash:
                raise ValueError(f"copied artifact hash mismatch: {content_hash}")
            entry["storage"] = "embedded"
            entry["bundle_path"] = f"artifacts/{content_hash}"
            entry["bytes"] = size
            entry.pop("uri", None)
            embedded_external += 1

        _write_json_file(stage / "artifact_index.json", artifact_index)
        file_hashes = {
            path.relative_to(stage).as_posix(): _sha256_file(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest.update(
            {
                "bundle_mode": "self_contained",
                "external_artifacts_embedded": embedded_external,
                "file_hashes": file_hashes,
                "integrity": {
                    "algorithm": "sha256",
                    "scope": "all bundle files except manifest.json",
                    "status": "verified",
                    "verified_file_count": len(file_hashes),
                },
            }
        )
        _write_manifest_with_total_size(stage, manifest)
        os.replace(stage, target)
        return target
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _trusted_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{label} must be a directory")
    return root


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _json_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _json_list(path: Path | None, label: str) -> list[dict[str, object]]:
    if path is None:
        raise ValueError(f"{label} is not declared")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array of objects")
    items = cast(list[object], value)
    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"{label} must be a JSON array of objects")
    return [cast(dict[str, object], item) for item in items]


def _verified_bundle_files(
    root: Path,
    manifest: dict[str, object],
    *,
    skip_paths: set[str] | None = None,
) -> dict[str, Path]:
    raw_hashes = manifest.get("file_hashes")
    if not isinstance(raw_hashes, dict):
        raise ValueError("bundle manifest has no file_hashes object")
    files: dict[str, Path] = {}
    for raw_relative, raw_hash in cast(dict[object, object], raw_hashes).items():
        if not isinstance(raw_relative, str) or not isinstance(raw_hash, str):
            raise ValueError("bundle manifest contains an invalid file hash entry")
        _validate_hash(raw_hash)
        relative = Path(raw_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("bundle manifest contains an unsafe file path")
        candidate = root / relative
        if candidate.is_symlink():
            raise ValueError(f"bundle file must not be a symlink: {raw_relative}")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"bundle file escapes its root: {raw_relative}") from exc
        if not resolved.is_file() or (
            raw_relative not in (skip_paths or set())
            and _sha256_file(resolved) != raw_hash
        ):
            raise ValueError(f"bundle file hash mismatch: {raw_relative}")
        files[raw_relative] = resolved
    return files


def _verified_artifact_index(
    root: Path, manifest: dict[str, object]
) -> tuple[Path, set[str]]:
    raw_hashes = manifest.get("file_hashes")
    if not isinstance(raw_hashes, dict):
        raise ValueError("bundle manifest has no file_hashes object")
    hashes = cast(dict[object, object], raw_hashes)
    raw_hash = hashes.get("artifact_index.json")
    if not isinstance(raw_hash, str):
        raise ValueError("bundle manifest does not declare artifact_index.json")
    _validate_hash(raw_hash)
    index_path = root / "artifact_index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("bundle artifact index must be a regular file")
    if index_path.resolve(strict=True).parent != root:
        raise ValueError("bundle artifact index escapes its root")
    if _sha256_file(index_path) != raw_hash:
        raise ValueError("bundle file hash mismatch: artifact_index.json")
    declared_paths = {relative for relative in hashes if isinstance(relative, str)}
    return index_path, declared_paths


def _external_artifacts(root: Path) -> dict[str, dict[str, object]]:
    manifest = _json_object(root / "manifest.json", "external manifest")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError("external manifest has no artifacts array")
    artifacts: dict[str, dict[str, object]] = {}
    for raw in cast(list[object], raw_artifacts):
        if not isinstance(raw, dict):
            raise ValueError("external manifest artifact must be an object")
        entry = cast(dict[str, object], raw)
        content_hash = entry.get("sha256")
        if not isinstance(content_hash, str):
            raise ValueError("external manifest artifact has no SHA-256")
        _validate_hash(content_hash)
        if content_hash in artifacts:
            raise ValueError("external manifest has duplicate content hashes")
        artifacts[content_hash] = entry
    return artifacts


def _reject_evaluation_only(artifact_index: list[dict[str, object]]) -> None:
    for entry in artifact_index:
        visibility = entry.get("visibility")
        if visibility not in {"model_readable", "audit_only", "evaluation_only"}:
            raise ValueError("artifact index requires an explicit valid visibility")
        if visibility == "evaluation_only":
            raise ValueError("self-contained export refuses evaluation_only artifacts")


def _artifact_body_paths(artifact_index: list[dict[str, object]]) -> set[str]:
    paths: set[str] = set()
    for entry in artifact_index:
        content_hash = entry.get("content_hash")
        if not isinstance(content_hash, str):
            raise ValueError("artifact index content_hash must be a string")
        paths.add(f"artifacts/{_validate_hash(content_hash)}")
        bundle_path = entry.get("bundle_path")
        if isinstance(bundle_path, str):
            paths.add(bundle_path)
    return paths


def _domain_artifact_visibilities(
    files: dict[str, Path], artifact_index_path: Path
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for path in files.values():
        if path == artifact_index_path or path.suffix not in {".json", ".jsonl"}:
            continue
        values = (
            [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if path.suffix == ".jsonl"
            else [json.loads(path.read_text(encoding="utf-8"))]
        )
        for value in values:
            _collect_artifact_visibilities(value, result)
    return result


def _collect_artifact_visibilities(value: object, result: dict[str, set[str]]) -> None:
    if isinstance(value, dict):
        item = cast(dict[object, object], value)
        content_hash = item.get("content_hash")
        visibility = item.get("visibility", "model_readable")
        if isinstance(content_hash, str) and isinstance(visibility, str):
            result.setdefault(content_hash, set()).add(visibility)
        for field in ("model_input_ref", "model_response_ref"):
            reference = item.get(field)
            if isinstance(reference, str):
                result.setdefault(reference, set()).add("model_readable")
        for nested in item.values():
            _collect_artifact_visibilities(nested, result)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _collect_artifact_visibilities(nested, result)


def _validate_index_visibilities(
    artifact_index: list[dict[str, object]],
    domain_visibilities: dict[str, set[str]],
) -> None:
    indexed: dict[str, str] = {}
    for entry in artifact_index:
        content_hash = entry.get("content_hash")
        visibility = entry.get("visibility")
        if not isinstance(content_hash, str) or not isinstance(visibility, str):
            raise ValueError("artifact index visibility metadata is invalid")
        if content_hash in indexed:
            raise ValueError("artifact index has duplicate content hashes")
        indexed[content_hash] = visibility
    for content_hash, visibilities in domain_visibilities.items():
        if content_hash not in indexed:
            raise ValueError(f"unregistered artifact reference: {content_hash}")
        merged = _merged_artifact_visibility(visibilities)
        if merged != indexed[content_hash]:
            raise ValueError(
                f"artifact visibility mismatch for {content_hash}: "
                f"index={indexed[content_hash]}, records={merged}"
            )


def _validate_embedded_layout(
    declared_paths: set[str], artifact_index: list[dict[str, object]]
) -> None:
    for entry in artifact_index:
        if entry.get("storage") != "embedded":
            continue
        content_hash = entry.get("content_hash")
        if not isinstance(content_hash, str):
            raise ValueError("artifact index content_hash must be a string")
        expected = f"artifacts/{_validate_hash(content_hash)}"
        if entry.get("bundle_path") != expected or expected not in declared_paths:
            raise ValueError(f"embedded artifact must use physical path {expected}")


def _verified_external_artifact(
    root: Path,
    artifacts: dict[str, dict[str, object]],
    content_hash: str,
) -> tuple[Path, int]:
    entry = artifacts.get(content_hash)
    if entry is None:
        raise ValueError(f"external artifact is absent from manifest: {content_hash}")
    raw_path = entry.get("path")
    raw_size = entry.get("bytes")
    if entry.get("visibility") == "evaluation_only":
        raise ValueError("self-contained export refuses evaluation_only artifacts")
    if (
        not isinstance(raw_path, str)
        or isinstance(raw_size, bool)
        or not isinstance(raw_size, int)
    ):
        raise ValueError(f"external artifact manifest entry is invalid: {content_hash}")
    relative = Path(raw_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.name.lower() in {"query.csv", "record.csv"}
    ):
        raise ValueError(f"unsafe external artifact path: {raw_path}")
    source = root / relative
    if source.is_symlink():
        raise ValueError(f"external artifact must not be a symlink: {raw_path}")
    resolved = source.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"external artifact escapes its root: {raw_path}") from exc
    if not resolved.is_file() or resolved.stat().st_size != raw_size:
        raise ValueError(f"external artifact size mismatch: {raw_path}")
    if _sha256_file(resolved) != content_hash:
        raise ValueError(f"external artifact hash mismatch: {raw_path}")
    return resolved, raw_size


def _write_manifest_with_total_size(root: Path, manifest: dict[str, object]) -> None:
    manifest_path = root / "manifest.json"
    manifest["bundle_size_bytes"] = 0
    for _attempt in range(4):
        _write_json_file(manifest_path, manifest)
        size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if manifest["bundle_size_bytes"] == size:
            return
        manifest["bundle_size_bytes"] = size
    _write_json_file(manifest_path, manifest)
    actual = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if manifest["bundle_size_bytes"] != actual:
        raise RuntimeError("bundle size did not stabilize")

"""Reproducible runtime version metadata for source and wheel installs."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

ROOTWALK_VERSION = "0.2.7"
PROMPT_VERSION = "m124"
TOOL_VERSION = "openrca-public-input-v82"
TERMINAL_POLICY_VERSION = 60
UV_LOCK_SHA256 = "2f4f93efaf2db5874ebcc24379208c218e48134a543507fbec9917d01d76b032"
DIRECT_DEPENDENCIES = (
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "opentelemetry-api",
    "opentelemetry-exporter-otlp-proto-http",
    "opentelemetry-sdk",
    "pydantic",
    "typer",
    "volcengine-python-sdk",
)


def source_tree_revision(root: Path | None = None) -> str:
    package = (root or Path(__file__).resolve().parent).resolve(strict=True)
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def installed_source_revision() -> str:
    return source_tree_revision()


@lru_cache(maxsize=1)
def installed_uv_lock_sha256() -> str:
    package = Path(__file__).resolve().parent
    checkout_lock = package.parent.parent / "uv.lock"
    if package.parent.name == "src" and checkout_lock.is_file():
        actual = hashlib.sha256(checkout_lock.read_bytes()).hexdigest()
        if actual != UV_LOCK_SHA256:
            raise RuntimeError("uv.lock changed without updating UV_LOCK_SHA256")
    return UV_LOCK_SHA256


@lru_cache(maxsize=1)
def installed_direct_dependency_versions() -> dict[str, str]:
    return {name: version(name) for name in DIRECT_DEPENDENCIES}


def _dependency_fingerprint(versions: dict[str, str]) -> str:
    encoded = json.dumps(versions, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def runtime_version_manifest() -> dict[str, object]:
    dependencies = dict(installed_direct_dependency_versions())
    return {
        "rootwalk_version": ROOTWALK_VERSION,
        "prompt_version": PROMPT_VERSION,
        "tool_version": TOOL_VERSION,
        "terminal_policy_version": TERMINAL_POLICY_VERSION,
        "code_revision": installed_source_revision(),
        "uv_lock_sha256": installed_uv_lock_sha256(),
        "direct_dependency_versions": dependencies,
        "direct_dependency_fingerprint": _dependency_fingerprint(dependencies),
    }

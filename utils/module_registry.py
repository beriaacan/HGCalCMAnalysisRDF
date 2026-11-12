#!/usr/bin/env python3
"""Persistent, append-only module-name to integer-ID registry."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import tempfile
from typing import Iterable, Mapping


REGISTRY_FORMAT = "hgcal_cm_module_registry"
SCHEMA_VERSION = 1
UNKNOWN_MODULE_ID = 0
DEFAULT_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "module_registry.json"
)


def _empty_registry() -> dict:
    return {
        "format": REGISTRY_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "revision": 0,
        "unknown_module_id": UNKNOWN_MODULE_ID,
        "next_id": UNKNOWN_MODULE_ID + 1,
        "modules": {},
    }


def _canonical_core(payload: Mapping) -> dict:
    return {
        "format": payload["format"],
        "schema_version": int(payload["schema_version"]),
        "revision": int(payload["revision"]),
        "unknown_module_id": int(payload["unknown_module_id"]),
        "next_id": int(payload["next_id"]),
        "modules": {
            str(name): int(module_id)
            for name, module_id in sorted(payload["modules"].items())
        },
    }


def registry_digest(payload: Mapping) -> str:
    encoded = json.dumps(
        _canonical_core(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_registry(payload: Mapping) -> dict:
    if payload.get("format") != REGISTRY_FORMAT:
        raise ValueError(
            f"Unsupported module registry format {payload.get('format')!r}; "
            f"expected {REGISTRY_FORMAT!r}."
        )
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported module registry schema_version "
            f"{payload.get('schema_version')!r}; expected {SCHEMA_VERSION}."
        )
    modules = payload.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("Module registry 'modules' must be a JSON object.")

    normalized = _canonical_core(payload)
    if normalized["unknown_module_id"] != UNKNOWN_MODULE_ID:
        raise ValueError(
            f"Module registry reserves ID {UNKNOWN_MODULE_ID} for UNKNOWN."
        )
    ids = list(normalized["modules"].values())
    if any(module_id <= UNKNOWN_MODULE_ID for module_id in ids):
        raise ValueError("Real module IDs must be positive integers.")
    if len(ids) != len(set(ids)):
        raise ValueError("Module registry contains duplicate integer IDs.")
    expected_next = max(ids, default=UNKNOWN_MODULE_ID) + 1
    if normalized["next_id"] != expected_next:
        raise ValueError(
            f"Invalid module registry next_id={normalized['next_id']}; "
            f"expected {expected_next}."
        )
    if normalized["revision"] < 0:
        raise ValueError("Module registry revision must be non-negative.")
    computed_digest = registry_digest(normalized)
    provided_digest = payload.get("digest_sha256")
    if provided_digest and provided_digest != computed_digest:
        raise ValueError(
            "Module registry digest mismatch; the mapping may have been edited "
            "without updating its versioned metadata."
        )
    normalized["digest_sha256"] = computed_digest
    return normalized


def load_registry(path: str = DEFAULT_REGISTRY_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Persistent module registry not found: {path}. Refusing to create a "
            "replacement implicitly because that could reuse permanent IDs."
        )
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_registry(payload)


def _write_registry(path: str, payload: Mapping) -> None:
    validated = validate_registry(payload)
    output = dict(validated)
    output["updated_utc"] = datetime.now(timezone.utc).isoformat()
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix="module_registry_", suffix=".json", dir=output_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@contextmanager
def _registry_lock(path: str):
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def ensure_modules(
    module_names: Iterable[str],
    path: str = DEFAULT_REGISTRY_PATH,
    *,
    create_registry: bool = False,
) -> dict:
    """Append unseen modules and return a validated registry snapshot.

    Existing entries are never renumbered or removed. New IDs are allocated in
    the order in which names are first supplied.
    """
    names = [str(name).strip() for name in module_names]
    if any(not name for name in names):
        raise ValueError("Module registry names must be non-empty strings.")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate module names supplied to registry: {names}")

    with _registry_lock(path):
        payload = (
            validate_registry(_empty_registry())
            if create_registry and not os.path.exists(path)
            else load_registry(path)
        )
        modules = dict(payload["modules"])
        new_names = [name for name in names if name not in modules]
        if new_names:
            next_id = int(payload["next_id"])
            for name in new_names:
                modules[name] = next_id
                next_id += 1
            payload.update(
                {
                    "modules": modules,
                    "next_id": next_id,
                    "revision": int(payload["revision"]) + 1,
                }
            )
            payload.pop("digest_sha256", None)
            _write_registry(path, payload)
            payload = load_registry(path)
            print(
                f"Module registry revision {payload['revision']}: assigned "
                + ", ".join(f"{name}={payload['modules'][name]}" for name in new_names)
            )
        return deepcopy(payload)


def preview_modules(
    module_names: Iterable[str], path: str = DEFAULT_REGISTRY_PATH
) -> dict:
    """Return the registry that would result from an append, without writing it."""
    names = [str(name).strip() for name in module_names]
    if any(not name for name in names):
        raise ValueError("Module registry names must be non-empty strings.")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate module names supplied to registry: {names}")
    payload = load_registry(path)
    modules = dict(payload["modules"])
    new_names = [name for name in names if name not in modules]
    if new_names:
        next_id = int(payload["next_id"])
        for name in new_names:
            modules[name] = next_id
            next_id += 1
        payload.update(
            {
                "modules": modules,
                "next_id": next_id,
                "revision": int(payload["revision"]) + 1,
            }
        )
        payload.pop("digest_sha256", None)
        payload = validate_registry(payload)
    return deepcopy(payload)


def module_ids(
    module_names: Iterable[str],
    *,
    registry: Mapping | None = None,
    path: str = DEFAULT_REGISTRY_PATH,
) -> dict[str, int]:
    payload = load_registry(path) if registry is None else validate_registry(registry)
    names = [str(name) for name in module_names]
    missing = [name for name in names if name not in payload["modules"]]
    if missing:
        raise KeyError(
            "Module(s) are absent from the persistent module registry: "
            f"{missing}. Register them while preparing new DNN inputs."
        )
    return {name: int(payload["modules"][name]) for name in names}


def snapshot_for_model(registry: Mapping, training_modules: Iterable[str]) -> dict:
    payload = validate_registry(registry)
    names = [str(name) for name in training_modules]
    ids = module_ids(names, registry=payload)
    snapshot = dict(payload)
    snapshot["training_modules"] = names
    snapshot["training_module_ids"] = ids
    snapshot["unknown_module_policy"] = "error"
    return snapshot


def write_snapshot(path: str, registry: Mapping, training_modules: Iterable[str]) -> dict:
    snapshot = snapshot_for_model(registry, training_modules)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return snapshot


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Append stable module IDs to the RDF-owned registry")
    parser.add_argument('--registry', required=True, help="Existing RDF registry JSON; never created implicitly")
    parser.add_argument('--add', nargs='+', required=True, help="Module names to append without renumbering")
    args = parser.parse_args()
    payload = ensure_modules(args.add, path=args.registry)
    for name, index in module_ids(args.add, registry=payload).items():
        print(f'{name}: {index}')


if __name__ == '__main__':
    main()

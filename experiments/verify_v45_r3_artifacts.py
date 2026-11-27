"""Verify v45 R3 public-benchmark artifact identity; no data are loaded.

Only checks source metadata and SHA256. Passing does NOT mean results reproduce.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

DATASETS = ("CIC-IDS2017", "CSE-CIC-IDS2018", "UNSW-NB15")
ROLES = ("raw_data", "preprocessing", "split_manifest", "configuration",
         "run_log", "recall_table4", "calibration", "ablation", "statistics")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(manifest_path: Path) -> list[str]:
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"Cannot read manifest: {exc}"]
    problems = []
    if manifest.get("manuscript_version") != "v45 R3":
        problems.append("manuscript_version must be 'v45 R3'")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        return problems + ["datasets must be an object keyed by dataset"]
    if set(datasets) != set(DATASETS):
        problems.append("datasets must be exactly: " + ", ".join(DATASETS))
    for name in DATASETS:
        data = datasets.get(name)
        if not isinstance(data, dict):
            problems.append(f"{name}: missing dataset object")
            continue
        url = data.get("official_source")
        if not isinstance(url, str) or not url.startswith("https://"):
            problems.append(f"{name}: official HTTPS source required")
        seeds = data.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(type(n) is not int for n in seeds):
            problems.append(f"{name}: integer seed list required")
        artifacts = data.get("artifacts")
        if not isinstance(artifacts, dict):
            problems.append(f"{name}: missing artifacts object")
            continue
        for role in ROLES:
            item = artifacts.get(role)
            if not isinstance(item, dict):
                problems.append(f"{name}: missing artifact {role}")
                continue
            rel, sha = item.get("path"), item.get("sha256")
            if not isinstance(rel, str) or not rel:
                problems.append(f"{name}/{role}: missing relative path")
                continue
            target = (root / rel).resolve()
            if not target.is_relative_to(root):
                problems.append(f"{name}/{role}: forbidden path traversal")
                continue
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                problems.append(f"{name}/{role}: invalid lowercase SHA256")
                continue
            if not target.is_file():
                problems.append(f"{name}/{role}: missing file {rel}")
            elif sha256_file(target) != sha:
                problems.append(f"{name}/{role}: checksum mismatch {rel}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    errors = validate_manifest(parser.parse_args().manifest)
    for error in errors:
        print("FAIL:", error)
    if errors:
        print("NOT VERIFIED: provenance artifacts missing or inconsistent")
        return 1
    print("PASS: files/hashes validated; experimental reproduction NOT certified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

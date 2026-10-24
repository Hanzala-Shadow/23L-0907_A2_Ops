"""Create stratified, duplicate-free split lists without copying images."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"real": 0, "ai_generated": 1}
SPLITS = ("train", "validation", "test")
COLUMNS = ("relative_path", "class", "label", "sha256")
GIB = 1024**3


def require_space(path, reserve_gib=2.0):
    free = shutil.disk_usage(path).free
    if free < (reserve_gib + 0.125) * GIB:
        raise RuntimeError("The split operation would encroach on the disk reserve.")
    return free


def proportions(config):
    values = {name: Decimal(str(config[name])) for name in SPLITS}
    if any(value <= 0 for value in values.values()) or sum(values.values()) != 1:
        raise ValueError("Split proportions must be positive and sum to one.")
    return values


def allocate_counts(total, ratios):
    """Use largest remainders; ties prefer test, then validation, then train."""
    quotas = {name: total * ratios[name] for name in SPLITS}
    counts = {
        name: int(quotas[name].to_integral_value(rounding=ROUND_FLOOR))
        for name in SPLITS
    }
    tie_priority = {"test": 2, "validation": 1, "train": 0}
    order = sorted(
        SPLITS,
        key=lambda name: (quotas[name] - counts[name], tie_priority[name]),
        reverse=True,
    )
    for name in order[: total - sum(counts.values())]:
        counts[name] += 1
    if min(counts.values()) == 0:
        raise ValueError("A class is too small to appear in all three splits.")
    return counts


def deduplicate(rows):
    groups = defaultdict(list)
    paths = set()
    for row in rows:
        if row["relative_path"] in paths:
            raise ValueError("The audit manifest contains a repeated file path.")
        paths.add(row["relative_path"])
        if row["status"] != "readable" or row["class"] not in LABELS:
            raise ValueError(
                "Unreadable files or unknown labels need review before splitting."
            )
        groups[row["sha256"]].append(row)
    selected, excluded = [], []
    for digest, group in sorted(groups.items()):
        if len({row["class"] for row in group}) != 1:
            raise ValueError(
                "An exact duplicate has conflicting labels; review it first."
            )
        ordered = sorted(group, key=lambda row: row["relative_path"])
        canonical = ordered[0]
        selected.append(canonical)
        for duplicate in ordered[1:]:
            excluded.append(
                {
                    "excluded_relative_path": duplicate["relative_path"],
                    "retained_relative_path": canonical["relative_path"],
                    "class": duplicate["class"],
                    "sha256": digest,
                    "reason": "exact_duplicate",
                }
            )
    return selected, sorted(excluded, key=lambda row: row["excluded_relative_path"])


def make_splits(rows, config):
    ratios = proportions(config)
    if not isinstance(config["seed"], int):
        raise TypeError("The split seed must be an integer.")
    selected, excluded = deduplicate(rows)
    result = {name: [] for name in SPLITS}
    for label in sorted(LABELS):
        members = [row for row in selected if row["class"] == label]
        # Hash ranking is stable across Python versions and manifest row order.
        members.sort(
            key=lambda row: (
                hashlib.sha256(f"{config['seed']}\0{row['sha256']}".encode()).digest(),
                row["relative_path"],
            )
        )
        counts = allocate_counts(len(members), ratios)
        start = 0
        for name in SPLITS:
            for row in members[start : start + counts[name]]:
                result[name].append(
                    {key: row[key] for key in ("relative_path", "class", "sha256")}
                    | {"label": LABELS[row["class"]]}
                )
            start += counts[name]
    for members in result.values():
        members.sort(key=lambda row: row["relative_path"])
    verify_splits(result, selected)
    return result, excluded


def verify_splits(splits, selected):
    expected = {row["sha256"] for row in selected}
    seen_hashes, seen_paths = set(), set()
    for name in SPLITS:
        if {row["class"] for row in splits[name]} != set(LABELS):
            raise ValueError(f"{name} is missing a class.")
        for row in splits[name]:
            if row["sha256"] in seen_hashes or row["relative_path"] in seen_paths:
                raise ValueError("Split overlap detected by path or content hash.")
            if row["label"] != LABELS[row["class"]]:
                raise ValueError("Inconsistent numeric label.")
            seen_hashes.add(row["sha256"])
            seen_paths.add(row["relative_path"])
    if seen_hashes != expected:
        raise ValueError("Split coverage does not match the unique dataset.")


def verify_source(rows, raw):
    fingerprint = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["relative_path"]):
        relative = PurePosixPath(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("Unsafe path in audit manifest.")
        path = raw.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(raw.resolve()) or path.is_symlink():
            raise ValueError("Manifest path leaves the raw dataset.")
        if path.stat().st_size != int(row["file_size_bytes"]):
            raise ValueError(f"File size changed since the audit: {relative}")
        with path.open("rb") as image:
            digest = hashlib.file_digest(image, "sha256").hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"File contents changed since the audit: {relative}")
        fingerprint.update(f"{relative}\0{row['class']}\0{digest}\n".encode())
    actual_paths = {
        path.relative_to(raw).as_posix() for path in raw.rglob("*") if path.is_file()
    }
    if actual_paths != {row["relative_path"] for row in rows}:
        raise ValueError("Raw file inventory differs from the audit manifest.")
    return fingerprint.hexdigest()


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=ROOT / "params.json")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "reports/data/image_manifest.csv"
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.reserve_gib < 2:
        parser.error("The agreed minimum reserve is 2 GiB.")
    if args.output_dir.resolve().is_relative_to(args.raw_dir.resolve()):
        parser.error("Split outputs must be outside the raw data directory.")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    require_space(args.output_dir.parent, args.reserve_gib)
    config = json.loads(args.params.read_text())["split"]
    with args.manifest.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    fingerprint = verify_source(rows, args.raw_dir)
    splits, excluded = make_splits(rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        write_csv(args.output_dir / f"{name}.csv", splits[name], COLUMNS)
    write_csv(
        args.output_dir / "excluded_duplicates.csv",
        excluded,
        (
            "excluded_relative_path",
            "retained_relative_path",
            "class",
            "sha256",
            "reason",
        ),
    )
    summary = {
        "schema_version": 1,
        "seed": config["seed"],
        "proportions": {name: float(config[name]) for name in SPLITS},
        "labels": LABELS,
        "ranking": "SHA-256(seed + NUL + image SHA-256), per class",
        "rounding": "largest remainders; ties prefer test, validation, train",
        "duplicate_policy": "retain lexicographically first path for each SHA-256",
        "input_files": len(rows),
        "selected_unique_images": sum(len(members) for members in splits.values()),
        "excluded_duplicate_files": len(excluded),
        "raw_dataset_fingerprint_sha256": fingerprint,
        "audit_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "splits": {
            name: {
                "total": len(splits[name]),
                "classes": dict(
                    sorted(Counter(row["class"] for row in splits[name]).items())
                ),
                "manifest_sha256": hashlib.sha256(
                    (args.output_dir / f"{name}.csv").read_bytes()
                ).hexdigest(),
            }
            for name in SPLITS
        },
        "overlap_by_path": 0,
        "overlap_by_sha256": 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(
        f"Free space: {require_space(args.output_dir, args.reserve_gib) / GIB:.2f} GiB"
    )


if __name__ == "__main__":
    main()

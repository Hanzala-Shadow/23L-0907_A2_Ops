"""Audit every raw file and generate reproducible metadata, statistics and EDA."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import statistics
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"AiArtData": "ai_generated", "RealArt": "real"}
LABELS = {"ai_generated": "AI-generated", "real": "Real (web-scraped)"}
COLORS = {"ai_generated": "#aa4499", "real": "#228833"}
GIB = 1024**3


def check_space(path, reserve_gib):
    free = shutil.disk_usage(path).free
    if free < (reserve_gib + 0.125) * GIB:
        raise RuntimeError(
            "Insufficient space for audit outputs above the disk reserve."
        )
    return free


def audit_rows(directory):
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Raw data contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        row = {
            "relative_path": relative.as_posix(),
            "class": CLASSES.get(relative.parts[0], "unknown"),
            "source_class_directory": relative.parts[0],
            "file_size_bytes": path.stat().st_size,
            "width": None,
            "height": None,
            "colour_mode": None,
            "format": None,
            "sha256": digest,
            "status": "readable",
            "error": "",
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    row.update(
                        width=image.width,
                        height=image.height,
                        colour_mode=image.mode,
                        format=image.format,
                    )
                    image.verify()
                # verify() alone can miss truncated pixel payloads; decode fully too.
                with Image.open(path) as image:
                    image.load()
        except (
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombWarning,
            Image.DecompressionBombError,
        ) as error:
            row["status"] = "unreadable"
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"Audited {len(rows)} files", flush=True)
    if not rows:
        raise ValueError("No files found in the dataset directory.")
    return rows


def numeric_stats(values):
    return {
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def format_number(value):
    if value is None:
        return "n/a"
    return f"{value:,.0f}" if value == int(value) else f"{value:,.1f}"


def summarize(rows):
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    groups = {digest: group for digest, group in by_hash.items() if len(group) > 1}
    for row in rows:
        group = groups.get(row["sha256"], [])
        row["duplicate_group_size"] = len(group) if group else 1
        row["is_exact_duplicate"] = bool(group)
        row["cross_class_duplicate"] = len({r["class"] for r in group}) > 1
    valid = [row for row in rows if row["status"] == "readable"]
    classes = {}
    for label in sorted({row["class"] for row in rows}):
        all_class = [row for row in rows if row["class"] == label]
        readable = [row for row in all_class if row["status"] == "readable"]
        classes[label] = {
            "total_files": len(all_class),
            "readable_images": len(readable),
            "unreadable_files": len(all_class) - len(readable),
            "total_bytes": sum(row["file_size_bytes"] for row in all_class),
            "width": numeric_stats([row["width"] for row in readable]),
            "height": numeric_stats([row["height"] for row in readable]),
            "file_size_bytes": numeric_stats(
                [row["file_size_bytes"] for row in all_class]
            ),
            "formats": dict(Counter(row["format"] for row in readable)),
            "colour_modes": dict(Counter(row["colour_mode"] for row in readable)),
            "duplicate_member_files": sum(
                row["is_exact_duplicate"] for row in all_class
            ),
            "landscape_images": sum(row["width"] > row["height"] for row in readable),
            "portrait_images": sum(row["width"] < row["height"] for row in readable),
            "square_images": sum(row["width"] == row["height"] for row in readable),
        }
    fingerprint = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["relative_path"]):
        fingerprint.update(
            f"{row['relative_path']}\0{row['class']}\0{row['sha256']}\n".encode()
        )
    return {
        "total_files": len(rows),
        "readable_images": len(valid),
        "unreadable_files": len(rows) - len(valid),
        "unknown_class_files": sum(row["class"] == "unknown" for row in rows),
        "total_bytes": sum(row["file_size_bytes"] for row in rows),
        "unique_file_hashes": len(by_hash),
        "readable_unique_hashes": len({row["sha256"] for row in valid}),
        "duplicate_groups": len(groups),
        "duplicate_member_files": sum(len(group) for group in groups.values()),
        "redundant_duplicate_files": sum(len(group) - 1 for group in groups.values()),
        "cross_class_duplicate_groups": sum(
            len({row["class"] for row in group}) > 1 for group in groups.values()
        ),
        "dataset_fingerprint_sha256": fingerprint.hexdigest(),
        "classes": classes,
    }


def write_csv(path, rows, columns=None):
    if columns is None:
        columns = list(rows[0])
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows, directory, output, seed):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports/setup/.matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [row for row in rows if row["status"] == "readable"]
    rng = random.Random(seed)
    classes = [label for label in LABELS if any(row["class"] == label for row in valid)]
    figure, axes = plt.subplots(
        len(classes), 6, figsize=(15, 3.5 * len(classes)), squeeze=False
    )
    samples = []
    for class_index, label in enumerate(classes):
        distinct = {}
        for row in valid:
            if row["class"] == label:
                distinct.setdefault(row["sha256"], row)
        selected = rng.sample(list(distinct.values()), min(6, len(distinct)))
        for index, axis in enumerate(axes[class_index]):
            axis.axis("off")
            if index >= len(selected):
                continue
            row = selected[index]
            with Image.open(directory / row["relative_path"]) as original:
                thumbnail = ImageOps.exif_transpose(original).convert("RGB")
                thumbnail.thumbnail((256, 256))
                canvas = Image.new("RGB", (256, 256), "#eeeeee")
                canvas.paste(
                    thumbnail,
                    ((256 - thumbnail.width) // 2, (256 - thumbnail.height) // 2),
                )
                axis.imshow(canvas)
            axis.set_title(
                f"{LABELS[label]}\n{row['width']} x {row['height']}", fontsize=10
            )
            samples.append(
                {
                    "sample_number": len(samples) + 1,
                    "class": label,
                    "relative_path": row["relative_path"],
                }
            )
    figure.suptitle(
        f"VisionGuard: six byte-distinct examples per class (seed {seed})", fontsize=15
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output / "sample_grid.png", dpi=140)
    plt.close(figure)
    write_csv(output / "sample_grid_index.csv", samples)

    for field, title, unit in [
        ("width", "Image width", "pixels"),
        ("height", "Image height", "pixels"),
        ("file_size_bytes", "Image file size", "KiB"),
    ]:
        all_values = np.array([row[field] for row in valid], dtype=float)
        if field == "file_size_bytes":
            all_values /= 1024
        log_scale = all_values.min() > 0 and all_values.max() / all_values.min() > 20
        low, high = all_values.min(), all_values.max()
        if low == high:
            low, high = low * 0.9, high * 1.1 + 1
        bins = np.geomspace(low, high, 31) if log_scale else np.linspace(low, high, 31)
        figure, axis = plt.subplots(figsize=(9, 4.8))
        for label in classes:
            values = np.array(
                [row[field] for row in valid if row["class"] == label], dtype=float
            )
            if field == "file_size_bytes":
                values /= 1024
            axis.hist(
                values,
                bins=bins,
                alpha=0.55,
                color=COLORS[label],
                label=f"{LABELS[label]} (n={len(values)})",
            )
            axis.axvline(
                np.median(values), color=COLORS[label], linestyle="--", linewidth=1.5
            )
        if log_scale:
            axis.set_xscale("log")
        axis.set(
            title=title,
            xlabel=f"{title} ({unit})" + ("; logarithmic scale" if log_scale else ""),
            ylabel="Number of images",
        )
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        figure.text(
            0.5,
            0.015,
            "Dashed lines mark class medians. Readable images only.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        figure.savefig(output / f"{field}_histogram.png", dpi=140)
        plt.close(figure)


def markdown_report(summary, output, seed):
    lines = [
        "# VisionGuard dataset audit",
        "",
        f"Audit time: {summary['timestamp']}. Raw files were inspected without modification.",
        "",
        "## Counts and integrity",
        "",
        "| Class | Files | Readable | Unreadable | Total MiB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, counts in summary["classes"].items():
        lines.append(
            f"| {LABELS.get(label, label)} | {counts['total_files']} | {counts['readable_images']} | {counts['unreadable_files']} | {counts['total_bytes'] / 1024**2:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                f"SHA-256 exact-duplicate groups: **{summary['duplicate_groups']}**; "
                f"files belonging to those groups: **{summary['duplicate_member_files']}**; "
                f"redundant copies beyond one file per hash: **{summary['redundant_duplicate_files']}**; "
                f"cross-class duplicate groups: **{summary['cross_class_duplicate_groups']}**."
            ),
            "",
            "Duplicate counts refer to identical file bytes, not visually similar images. Originals and duplicates remain intact in data/raw. Keep each duplicate group in a single split, or deduplicate before splitting; inspect conflicting labels before choosing a class.",
            "",
            "## Dimensions and file size",
            "",
            "| Class | Quantity | Minimum | Median | Maximum |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for label, counts in summary["classes"].items():
        for key, quantity in [
            ("width", "Width (px)"),
            ("height", "Height (px)"),
            ("file_size_bytes", "File size (bytes)"),
        ]:
            stats = counts[key]
            values = [format_number(stats[k]) for k in ("min", "median", "max")]
            lines.append(
                f"| {LABELS.get(label, label)} | {quantity} | {' | '.join(values)} |"
            )
    lines.extend(
        [
            "",
            "Dimension statistics and figures use readable images only; file-size statistics above include every audited file. Width and height are encoded dimensions before EXIF orientation handling. Every file has its original SHA-256, class, size, detected format/mode, readability status and duplicate flags in image_manifest.csv.",
            "",
            "## Visual exploration",
            "",
            "![Class samples](sample_grid.png)",
            "",
            f"Figure 1. Six readable, byte-distinct examples per class were sampled with seed {seed}; display thumbnails preserve aspect ratio and apply EXIF orientation. The real-labeled content includes illustrations and paintings as well as photographs, so the task is broader than distinguishing AI images from photographs alone. Labels come from source folders and are not independently verified ground truth.",
        ]
    )
    for number, key, title, advice in [
        (
            2,
            "width",
            "width",
            "A fixed training input width is necessary; choose resizing and padding/cropping deliberately to avoid unintended distortion.",
        ),
        (
            3,
            "height",
            "height",
            "Inspect aspect ratios as well as height before selecting the model preprocessing policy.",
        ),
        (
            4,
            "file_size_bytes",
            "file size",
            "File size is an encoding/storage property, not a reliable authenticity label, so the model should learn image content rather than file metadata.",
        ),
    ]:
        medians = "; ".join(
            f"{LABELS.get(label, label)}: {counts[key]['median']:g}"
            for label, counts in summary["classes"].items()
            if counts[key]["median"] is not None
        )
        unit = "bytes" if key == "file_size_bytes" else "pixels"
        lines.extend(
            [
                "",
                f"![{title.title()} distribution]({key}_histogram.png)",
                "",
                f"Figure {number}. The class {title} distributions have medians of {medians} ({unit}); the dashed lines mark those medians. {advice}",
            ]
        )
    lines.extend(["", "## Formats, colour modes and shapes", ""])
    for label, counts in summary["classes"].items():
        lines.append(
            f"- **{LABELS.get(label, label)}:** formats {json.dumps(counts['formats'])}; colour modes {json.dumps(counts['colour_modes'])}; landscape/portrait/square = {counts['landscape_images']}/{counts['portrait_images']}/{counts['square_images']}."
        )
    non_rgb = sum(
        count
        for counts in summary["classes"].values()
        for mode, count in counts["colour_modes"].items()
        if mode != "RGB"
    )
    lines.extend(
        [
            "",
            "## Findings that affect preparation",
            "",
            f"- **Class balance:** AI-generated content represents {100 * summary['classes']['ai_generated']['total_files'] / summary['total_files']:.1f}% of raw files. Preserve class proportions in the eventual splits and report precision/recall/F1 alongside accuracy.",
            f"- **Colour handling:** {non_rgb} images use palette, grayscale or RGBA modes. Convert inputs consistently to RGB, with an explicit alpha-compositing policy shared by training and serving.",
            "- **Resolution and aspect ratio:** image dimensions vary widely and include landscape, portrait and square images. Choose a bounded input size and document the aspect-ratio policy before training.",
            "- **Labels and scope:** the real-labeled folder includes photographs and artwork. Do not treat artistic appearance as proof of AI generation or present this small source dataset as independent verification of every label.",
            "- **Duplicate handling:** preserve the raw source and exclude redundant copies through a deterministic split manifest, or group duplicates into one split. Do not let identical bytes appear in both training and evaluation sets.",
        ]
    )
    lines.extend(
        [
            "",
            "## Reproducibility and storage",
            "",
            f"Dataset fingerprint: `{summary['dataset_fingerprint_sha256']}`. This hashes sorted relative paths, labels and each original file's SHA-256.",
            "",
            f"Free space after the audit: **{summary['free_after_bytes'] / GIB:.2f} GiB**. Required reserve: **{summary['required_reserve_bytes'] / GIB:.2f} GiB**. Download version, archive SHA-256, CRC verification and peak-space measurements are in dataset_source.json.",
            "",
            "The next step is to decide duplicate handling and make reproducible class-stratified train/validation/test lists. No split, training or original-file deletion was performed by this audit.",
            "",
        ]
    )
    (output / "audit.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/data")
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.reserve_gib < 2:
        parser.error("The agreed minimum reserve is 2 GiB.")
    if not args.data_dir.is_dir():
        parser.error("Dataset directory does not exist.")
    if args.output_dir.resolve().is_relative_to(args.data_dir.resolve()):
        parser.error("Audit outputs must be outside the raw dataset directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_free = check_space(args.output_dir, args.reserve_gib)
    rows = audit_rows(args.data_dir)
    summary = summarize(rows)
    if summary["unknown_class_files"]:
        raise ValueError(
            "Unexpected class directories found; inspect the manifest logic."
        )
    write_csv(args.output_dir / "image_manifest.csv", rows)
    class_rows = []
    for label, counts in summary["classes"].items():
        class_row = {"class": label}
        for key, value in counts.items():
            if isinstance(value, dict) and key in {
                "width",
                "height",
                "file_size_bytes",
            }:
                class_row.update(
                    {
                        f"{key}_{statistic}": number
                        for statistic, number in value.items()
                    }
                )
            else:
                class_row[key] = json.dumps(value) if isinstance(value, dict) else value
        class_rows.append(class_row)
    write_csv(args.output_dir / "class_summary.csv", class_rows)
    write_csv(
        args.output_dir / "unreadable_files.csv",
        [row for row in rows if row["status"] != "readable"],
        list(rows[0]),
    )
    write_csv(
        args.output_dir / "duplicate_files.csv",
        [row for row in rows if row["is_exact_duplicate"]],
        list(rows[0]),
    )
    make_figures(rows, args.data_dir, args.output_dir, args.seed)
    summary.update(
        timestamp=dt.datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
        seed=args.seed,
        raw_directory=(
            args.data_dir.resolve().relative_to(ROOT).as_posix()
            if args.data_dir.resolve().is_relative_to(ROOT)
            else str(args.data_dir.resolve())
        ),
        free_before_bytes=initial_free,
        free_after_bytes=check_space(args.output_dir, args.reserve_gib),
        required_reserve_bytes=int(args.reserve_gib * GIB),
        dimension_statistics="readable images only, before EXIF orientation",
        file_size_statistics="all audited files",
        raw_files_modified=False,
    )
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    markdown_report(summary, args.output_dir, args.seed)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Audit report: {args.output_dir / 'audit.md'}", flush=True)


if __name__ == "__main__":
    main()

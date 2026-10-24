"""Download and safely extract the assignment's versioned public Kaggle dataset."""

import argparse
import datetime as dt
import hashlib
import json
import shutil
import stat
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATASET = "cashbowman/ai-generated-images-vs-real-images"
API = "https://www.kaggle.com/api/v1/datasets"
GIB = 1024**3
CHUNK = 1024**2


class DiskBudget:
    def __init__(self, root, reserve_gib=2.0):
        self.root = root
        self.reserve = int(reserve_gib * GIB)
        self.margin = 128 * 1024**2
        self.minimum_observed = shutil.disk_usage(root).free

    def check(self, additional=0):
        free = shutil.disk_usage(self.root).free
        self.minimum_observed = min(self.minimum_observed, free)
        if free - additional < self.reserve + self.margin:
            raise RuntimeError(
                f"Disk budget: {free / GIB:.2f} GiB free; operation needs "
                f"{additional / GIB:.2f} GiB plus the reserve and safety margin."
            )
        return free


def request(url):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "VisionGuard-assignment-audit/1.0"}
        ),
        timeout=60,
    )


def safe_members(archive):
    """Reject traversal, symlinks, encrypted entries and duplicate ZIP paths."""
    members = archive.infolist()
    seen = set()
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
            raise ValueError(f"Unsafe archive path: {member.filename!r}")
        if not path.parts or path.parts[0] not in {"AiArtData", "RealArt"}:
            raise ValueError(f"Unexpected dataset directory: {member.filename!r}")
        if member.filename in seen:
            raise ValueError(f"Duplicate archive path: {member.filename!r}")
        seen.add(member.filename)
        if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
            raise ValueError(f"Unsupported archive entry: {member.filename!r}")
    return members


def download(reserve_gib):
    budget = DiskBudget(ROOT, reserve_gib)
    initial_free = budget.check()
    destination = ROOT / "data/raw"
    if destination.exists():
        raise FileExistsError(
            "data/raw already exists; audit it instead of downloading again."
        )
    with request(f"{API}/view/{DATASET}") as response:
        metadata = json.load(response)
    version = int(metadata["currentVersionNumber"])
    expected_size = int(metadata["totalBytes"])
    budget.check(additional=2 * expected_size + 64 * 1024**2)
    print(f"Dataset: {DATASET}, version {version}", flush=True)
    print(f"Listed data size: {expected_size / 1024**2:.1f} MiB", flush=True)

    downloads = ROOT / "data/.downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive_path = downloads / f"kaggle-v{version}.zip.part"
    staging = ROOT / "data/.extracting"
    if archive_path.exists() or staging.exists():
        raise FileExistsError(
            "Partial download/extraction exists; inspect it before retrying."
        )
    archive_hash = hashlib.sha256()
    downloaded = 0
    last_notice = time.monotonic()
    url = f"{API}/download/{DATASET}?datasetVersionNumber={version}"
    try:
        with request(url) as response, archive_path.open("xb") as output:
            length = response.headers.get("Content-Length")
            expected_archive_size = int(length) if length else None
            if expected_archive_size is not None:
                budget.check(additional=expected_archive_size)
            while chunk := response.read(CHUNK):
                if downloaded == 0 and not chunk.startswith(b"PK\x03\x04"):
                    raise ValueError(
                        "Kaggle returned a non-ZIP response; download stopped."
                    )
                budget.check(additional=len(chunk))
                output.write(chunk)
                archive_hash.update(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_notice >= 15:
                    print(
                        f"Downloaded {downloaded / 1024**2:.1f} MiB; "
                        f"free {budget.check() / GIB:.2f} GiB",
                        flush=True,
                    )
                    last_notice = time.monotonic()
            if (
                expected_archive_size is not None
                and downloaded != expected_archive_size
            ):
                raise ValueError("Downloaded size does not match Content-Length.")

        extracted_bytes = 0
        files = 0
        with zipfile.ZipFile(archive_path) as archive:
            members = safe_members(archive)
            total_uncompressed = sum(m.file_size for m in members if not m.is_dir())
            budget.check(additional=total_uncompressed + 64 * 1024**2)
            print(
                f"ZIP manifest: {len(members)} entries, "
                f"{total_uncompressed / 1024**2:.1f} MiB uncompressed.",
                flush=True,
            )
            staging.mkdir()
            for member in members:
                target = staging.joinpath(*PurePosixPath(member.filename).parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                # Reading to EOF checks each ZIP member's CRC as well as its size.
                with archive.open(member) as source, target.open("xb") as output:
                    while chunk := source.read(CHUNK):
                        budget.check(additional=len(chunk))
                        output.write(chunk)
                        written += len(chunk)
                if written != member.file_size:
                    raise ValueError(f"Extracted size mismatch: {member.filename}")
                extracted_bytes += written
                files += 1
            if extracted_bytes != total_uncompressed:
                raise ValueError("Extracted total differs from the ZIP manifest.")
        if sum(p.is_file() for p in staging.rglob("*")) != files:
            raise ValueError("Extracted file count differs from the ZIP manifest.")
        staging.rename(destination)
        archive_path.unlink()
        downloads.rmdir()
        report = {
            "timestamp": dt.datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
            "dataset": DATASET,
            "source_url": f"https://www.kaggle.com/datasets/{DATASET}",
            "download_api": url,
            "version": version,
            "last_updated": metadata.get("lastUpdated"),
            "license": metadata.get("licenseName"),
            "listed_data_bytes": expected_size,
            "archive_bytes": downloaded,
            "archive_sha256": archive_hash.hexdigest(),
            "extracted_bytes": extracted_bytes,
            "extracted_files": files,
            "archive_crc_checks": "passed for every extracted member",
            "archive_removed_after_verified_extraction": True,
            "raw_directory": "data/raw",
            "free_before_bytes": initial_free,
            "free_after_bytes": budget.check(),
            "minimum_observed_free_bytes": budget.minimum_observed,
            "required_reserve_bytes": budget.reserve,
            "safety_margin_bytes": budget.margin,
        }
        report_path = ROOT / "reports/data/dataset_source.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    except BaseException:
        # These paths were created by this invocation; never delete existing raw data.
        if archive_path.exists():
            archive_path.unlink()
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.reserve_gib < 2:
        parser.error("The agreed minimum reserve is 2 GiB.")
    download(args.reserve_gib)

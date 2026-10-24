"""Exercise integrity cases that would otherwise cause data leakage or bad counts."""

import io
import shutil
import stat
import zipfile

import pytest
from PIL import Image

from scripts.audit_dataset import audit_rows, summarize
from scripts.download_dataset import DiskBudget, safe_members


def test_audit_finds_corruption_modes_and_cross_class_duplicates(tmp_path):
    ai = tmp_path / "AiArtData/nested"
    real = tmp_path / "RealArt"
    ai.mkdir(parents=True)
    real.mkdir()
    Image.new("RGB", (4, 6), "red").save(ai / "original.png")
    shutil.copyfile(ai / "original.png", ai / "copy.png")
    shutil.copyfile(ai / "original.png", real / "conflicting-label.png")
    Image.new("L", (2, 5), 17).save(real / "grayscale.png")
    (real / "broken.jpg").write_bytes(b"not an image")

    rows = audit_rows(tmp_path)
    summary = summarize(rows)
    assert summary["total_files"] == 5
    assert summary["readable_images"] == 4
    assert summary["unreadable_files"] == 1
    assert summary["readable_unique_hashes"] == 2
    assert summary["duplicate_groups"] == 1
    assert summary["duplicate_member_files"] == 3
    assert summary["redundant_duplicate_files"] == 2
    assert summary["cross_class_duplicate_groups"] == 1
    assert summary["classes"]["ai_generated"]["total_files"] == 2
    assert summary["classes"]["real"]["colour_modes"] == {"RGB": 1, "L": 1}
    grayscale = next(
        row for row in rows if row["relative_path"].endswith("grayscale.png")
    )
    assert (grayscale["width"], grayscale["height"], grayscale["format"]) == (
        2,
        5,
        "PNG",
    )
    broken = next(row for row in rows if row["relative_path"].endswith("broken.jpg"))
    assert broken["error"].startswith("UnidentifiedImageError:")
    assert len(broken["sha256"]) == 64
    assert (
        summary["dataset_fingerprint_sha256"]
        == summarize(list(reversed(rows)))["dataset_fingerprint_sha256"]
    )


@pytest.mark.parametrize(
    "name",
    ["AiArtData/../../escape.jpg", "/AiArtData/escape.jpg", "unexpected/image.jpg"],
)
def test_archive_rejects_unsafe_or_unexpected_paths(name):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, b"payload")
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError):
        safe_members(archive)


def test_archive_rejects_symlinks():
    stream = io.BytesIO()
    entry = zipfile.ZipInfo("AiArtData/link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(entry, "/outside/file")
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError):
        safe_members(archive)


def test_disk_budget_reserves_space_before_writing(monkeypatch, tmp_path):
    disk_usage = type(shutil.disk_usage(tmp_path))
    available = int(2.125 * 1024**3)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _: disk_usage(10 * 1024**3, 0, available)
    )
    budget = DiskBudget(tmp_path, reserve_gib=2)
    assert budget.check() == available
    with pytest.raises(RuntimeError, match="Disk budget"):
        budget.check(additional=1)

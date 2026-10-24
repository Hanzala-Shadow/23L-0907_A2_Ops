"""Check reproducibility, leakage prevention and source integrity."""

import copy
import hashlib

import pytest

from scripts.prepare_splits import make_splits, verify_source, verify_splits

CONFIG = {"seed": 42, "train": 0.7, "validation": 0.15, "test": 0.15}


def sample_rows():
    return [
        {
            "relative_path": f"{label}/{index:02}.png",
            "class": label,
            "sha256": hashlib.sha256(f"{label}/{index}".encode()).hexdigest(),
            "status": "readable",
        }
        for label in ("ai_generated", "real")
        for index in range(20)
    ]


def test_splits_are_reproducible_and_ignore_manifest_order():
    rows = sample_rows()
    first, excluded = make_splits(rows, CONFIG)
    second, _ = make_splits(list(reversed(rows)), CONFIG)
    assert first == second
    assert excluded == []
    assert sum(len(split) for split in first.values()) == len(rows)
    assert all(
        {row["class"] for row in split} == {"real", "ai_generated"}
        for split in first.values()
    )


def test_duplicate_exclusion_is_deterministic():
    rows = sample_rows()
    duplicate = rows[0] | {"relative_path": "ai_generated/z-copy.png"}
    splits, excluded = make_splits([duplicate, *rows], CONFIG)
    assert len(excluded) == 1
    assert excluded[0]["retained_relative_path"] == rows[0]["relative_path"]
    selected = [row for split in splits.values() for row in split]
    assert len(selected) == len(rows)
    assert duplicate["relative_path"] not in {row["relative_path"] for row in selected}


def test_conflicting_duplicate_labels_require_review():
    rows = sample_rows()
    conflict = rows[0] | {"relative_path": "real/conflict.png", "class": "real"}
    with pytest.raises(ValueError, match="conflicting labels"):
        make_splits([*rows, conflict], CONFIG)


def test_changed_seed_changes_membership():
    rows = sample_rows()
    first, _ = make_splits(rows, CONFIG)
    second, _ = make_splits(rows, CONFIG | {"seed": 43})
    assert first["test"] != second["test"]
    assert {key: len(value) for key, value in first.items()} == {
        key: len(value) for key, value in second.items()
    }


def test_overlap_is_rejected():
    rows = sample_rows()
    splits, _ = make_splits(rows, CONFIG)
    damaged = copy.deepcopy(splits)
    damaged["test"][0] = damaged["train"][0]
    with pytest.raises(ValueError, match="overlap"):
        verify_splits(damaged, rows)


def test_source_changes_since_audit_are_rejected(tmp_path):
    image = tmp_path / "example.png"
    image.write_bytes(b"abc")
    rows = [
        {
            "relative_path": image.name,
            "file_size_bytes": 3,
            "class": "real",
            "sha256": hashlib.sha256(b"abc").hexdigest(),
        }
    ]
    assert len(verify_source(rows, tmp_path)) == 64
    image.write_bytes(b"abd")
    with pytest.raises(ValueError, match="contents changed"):
        verify_source(rows, tmp_path)

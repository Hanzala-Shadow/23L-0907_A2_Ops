"""Check cost-based thresholds and enforcement of a single test pass."""

import json

import numpy as np
import pytest

from visionguard.evaluation import binary_metrics, choose_threshold, predict_once


def test_threshold_uses_error_cost_and_deterministic_ties():
    labels, probabilities = [0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9]
    selected, curve = choose_threshold(labels, probabilities, 2, 1)
    assert selected["weighted_errors"] == min(row["weighted_errors"] for row in curve)
    assert selected["fp"] == 0 and selected["fn"] == 1
    assert 0.6 < selected["threshold"] < 0.9
    assert binary_metrics(labels, probabilities, selected["threshold"])[
        "roc_auc"
    ] == pytest.approx(0.75)


@pytest.mark.parametrize("probabilities", [[0.1, float("nan")], [-0.1, 0.5], [0.1]])
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError):
        choose_threshold([0, 1], probabilities)


def test_saved_predictions_prevent_second_inference_and_bind_identity(tmp_path):
    calls = []
    identity = {"images": 2, "model_sha256": "same", "plan_sha256": "sealed"}

    def predict():
        calls.append(1)
        return [0.2, 0.8]

    first, fresh = predict_once(tmp_path, identity, predict)
    second, repeated = predict_once(tmp_path, identity, predict)
    assert fresh and not repeated and calls == [1]
    np.testing.assert_array_equal(first, second)
    with pytest.raises(RuntimeError, match="identity changed"):
        predict_once(tmp_path, {**identity, "plan_sha256": "changed"}, predict)
    path = tmp_path / "predictions.json"
    saved = json.loads(path.read_text())
    saved["probabilities"][0] = 0.3
    path.write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="integrity"):
        predict_once(tmp_path, identity, predict)
    assert calls == [1]


def test_interrupted_pass_is_not_repeated(tmp_path):
    identity = {"images": 2}

    def fail():
        raise OSError("interrupted")

    with pytest.raises(OSError):
        predict_once(tmp_path, identity, fail)
    with pytest.raises(RuntimeError, match="refusing to repeat"):
        predict_once(tmp_path, identity, lambda: pytest.fail("must not run"))

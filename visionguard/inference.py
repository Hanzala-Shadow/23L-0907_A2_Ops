"""Load a classifier once and apply its versioned decision policy."""

import numpy as np

from visionguard.preprocessing import POLICY_VERSION, preprocess_image


class ImagePredictor:
    def __init__(self, model, threshold, image_size=128):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be in [0, 1]")
        self.model = model
        self.threshold = threshold
        self.image_size = image_size

    def predict(self, source):
        pixels = preprocess_image(source, self.image_size)
        probability = float(
            np.asarray(self.model(pixels[None], training=False)).reshape(-1)[0]
        )
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability")
        is_ai = probability >= self.threshold
        return {
            "label": "ai_generated" if is_ai else "real",
            "ai_probability": probability,
            "confidence": probability if is_ai else 1 - probability,
            "threshold": self.threshold,
        }


def load_by_alias(name, alias, destination):
    import mlflow
    import mlflow.tensorflow

    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    if version.tags["preprocessing_policy"] != POLICY_VERSION:
        raise ValueError("Registry preprocessing policy is incompatible")
    # Resolve the alias once, then pin both weights and threshold to that version.
    model = mlflow.tensorflow.load_model(
        f"models:/{name}/{version.version}", dst_path=str(destination)
    )
    predictor = ImagePredictor(
        model,
        float(version.tags["decision_threshold"]),
        int(version.tags["image_size"]),
    )
    return predictor, version

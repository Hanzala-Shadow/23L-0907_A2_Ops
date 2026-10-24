"""Prepare generated inputs and a local-only Postman environment."""

import json
import os
from pathlib import Path

from PIL import Image


def main():
    output = Path("reports/training-runtime/postman")
    output.mkdir(parents=True, exist_ok=True)
    for color in ["red", "blue", "white"]:
        Image.new("RGB", (160, 96), color).save(output / (color + ".png"))
    (output / "invalid.txt").write_text(
        "Generated invalid upload for API verification\n"
    )
    key = os.environ["MODEL_API_KEY"]
    environment = {
        "name": "23L-0907_A2_Ops local secrets",
        "values": [{"key": "api_key", "value": key, "enabled": True, "type": "secret"}],
        "_postman_variable_scope": "environment",
    }
    path = output / "private.postman_environment.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(environment, stream)
    os.chmod(path, 0o600)


if __name__ == "__main__":
    main()

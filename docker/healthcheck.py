"""Probe readiness without importing TensorFlow or loading another model."""

import json
from urllib.request import urlopen


def main():
    with urlopen("http://127.0.0.1:8000/health", timeout=4) as response:
        status = json.load(response)
        if response.status != 200 or status["status"] != "ok":
            raise SystemExit(1)


if __name__ == "__main__":
    main()

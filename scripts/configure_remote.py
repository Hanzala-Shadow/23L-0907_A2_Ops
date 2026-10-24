"""Enable cloud inference without changing database or session credentials."""

import json
import sys
from pathlib import Path

from dotenv import dotenv_values, set_key

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visionguard.remote import RemoteModelService


def main():
    path = ROOT / ".env"
    if not path.exists():
        raise SystemExit(
            "Configure the local Flask app before enabling remote inference"
        )
    values = dotenv_values(path)
    cloud = dotenv_values(ROOT / ".env.cloud")
    policy_path = ROOT / values.get("MODEL_POLICY_PATH", "models/champion/policy.json")
    service = RemoteModelService(
        cloud.get("MODEL_API_URL", ""),
        cloud.get("MODEL_API_KEY", ""),
        policy_path,
        ROOT / "cloud/model-manifest.json",
    )
    policy = json.loads(policy_path.read_text())
    if float(values["DECISION_THRESHOLD"]) != policy["threshold"]:
        raise SystemExit("The local threshold differs from the saved champion")
    updates = {
        "INFERENCE_BACKEND": "remote",
        "MODEL_API_URL": service.url,
        "MODEL_API_KEY": cloud["MODEL_API_KEY"],
        "MODEL_API_CONNECT_TIMEOUT": "3",
        "MODEL_API_READ_TIMEOUT": "45",
        "MODEL_API_MANIFEST_PATH": "cloud/model-manifest.json",
        "COMPOSE_FILE": "compose.yaml:compose.cloud.yaml",
    }
    for name, value in updates.items():
        set_key(path, name, value)
    path.chmod(0o600)
    print(
        "Cloud inference configured. PostgreSQL and session credentials were preserved."
    )


if __name__ == "__main__":
    main()

"""Create local web secrets without replacing existing tracking credentials."""

import json
import secrets
from pathlib import Path

from dotenv import dotenv_values, set_key
from sqlalchemy import URL

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / ".env"
    if not path.exists():
        path.write_text((ROOT / ".env.example").read_text())
    path.chmod(0o600)
    values = dotenv_values(path)
    updates = {}
    defaults = {
        "POSTGRES_DB": "visionguard",
        "POSTGRES_USER": "visionguard",
        "POSTGRES_PORT": "5433",
        "APP_BIND": "127.0.0.1:5000",
        "MAX_UPLOAD_BYTES": "8388608",
        "MAX_IMAGE_PIXELS": "16000000",
        "DISK_RESERVE_GIB": "2",
        "SESSION_COOKIE_SECURE": "false",
        "MODEL_PATH": "models/champion/best.keras",
        "MODEL_POLICY_PATH": "models/champion/policy.json",
    }
    for name, default in defaults.items():
        if not values.get(name):
            updates[name] = default
    for name in ("FLASK_SECRET_KEY", "POSTGRES_PASSWORD"):
        if not values.get(name) or values[name].startswith("replace_"):
            updates[name] = secrets.token_urlsafe(48)
    configured = {**values, **updates}
    if (
        not values.get("DATABASE_URL")
        or "replace_with_a_local_secret" in values["DATABASE_URL"]
    ):
        updates["DATABASE_URL"] = URL.create(
            "postgresql+psycopg",
            username=configured["POSTGRES_USER"],
            password=configured["POSTGRES_PASSWORD"],
            host="127.0.0.1",
            port=int(configured["POSTGRES_PORT"]),
            database=configured["POSTGRES_DB"],
        ).render_as_string(hide_password=False)
    if not values.get("DECISION_THRESHOLD"):
        policy = json.loads((ROOT / configured["MODEL_POLICY_PATH"]).read_text())
        updates["DECISION_THRESHOLD"] = str(policy["threshold"])
    for name, value in updates.items():
        set_key(path, name, value)
    path.chmod(0o600)
    print("Local web settings are ready. Existing credentials were preserved.")


if __name__ == "__main__":
    main()

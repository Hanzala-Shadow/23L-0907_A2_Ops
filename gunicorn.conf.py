"""Small CPU deployment with one long-lived model instance."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

bind = os.environ.get("APP_BIND", "127.0.0.1:5000")
workers = 1
threads = 2
worker_class = "gthread"
timeout = 120
accesslog = "-"
errorlog = "-"
capture_output = True

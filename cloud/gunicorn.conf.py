"""One bounded inference worker for Render's free compute plan."""

import os

bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
workers = 1
threads = 2
timeout = 120
graceful_timeout = 30
worker_tmp_dir = "/dev/shm"
accesslog = "-"
errorlog = "-"
access_log_format = "%(h)s %(m)s %(U)s %(s)s %(L)s"

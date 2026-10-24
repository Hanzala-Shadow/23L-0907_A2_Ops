"""Gunicorn entry point for the standalone cloud API."""

from visionguard.cloud import create_app

app = create_app()

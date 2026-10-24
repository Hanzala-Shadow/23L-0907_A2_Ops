"""Gunicorn entry point; one model is loaded per worker at startup."""

from visionguard.web import create_app

app = create_app()

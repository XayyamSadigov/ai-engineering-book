# path: book/projects/p1-extraction-api/extraction_api/api/__init__.py
from .app import app_factory, create_app

__all__ = ["create_app", "app_factory"]

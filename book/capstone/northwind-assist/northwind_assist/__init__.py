# path: book/capstone/northwind-assist/northwind_assist/__init__.py
"""Northwind Assist: the capstone system of the AI Engineering book (Chapter 39)."""
from . import _paths

_paths.install()   # example modules on sys.path before any submodule imports them (idempotent)

__version__ = "1.0.0"

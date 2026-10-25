"""Resolve project files without depending on the enclosing folder's name."""
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def read_json(relative_path):
    import json
    return json.loads((project_root() / relative_path).read_text(encoding="utf-8"))

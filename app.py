"""Deployment entry point using the same application and launcher as local runs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.gradio_app import demo, launch_app  # noqa: E402, F401

if __name__ == "__main__":
    launch_app()

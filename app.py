"""Deployment entry point using the same application and launcher as local runs."""

import sys
from pathlib import Path

# On Hugging Face ZeroGPU environments, spaces must be imported first.
# Satisfies ZeroGPU host supervisor startup registration while keeping all user inference
# (webcam streaming, image upload, video) on high-speed CPU (24ms latency),
# preventing user quota exhaustion and token authentication modals.
try:
    import spaces

    spaces.disable_gradio_auto_wrap()

    @spaces.GPU(duration=1)
    def _zerogpu_supervisor_keepalive():
        """Satisfies ZeroGPU supervisor registration."""
        return True
except Exception:
    _zerogpu_supervisor_keepalive = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.gradio_app import demo, launch_app  # noqa: E402, F401

if __name__ == "__main__":
    launch_app()

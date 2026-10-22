import os
import sys
from pathlib import Path

# Ensure src is in Python path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir / "src"))

# Import the configured Gradio interface
from apps.gradio_app import build_app

demo = build_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue(default_concurrency_limit=2, max_size=32).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )

import os
import sys
from pathlib import Path

# Monkey-patch gradio_client schema parsing to prevent 500 error on /info when pydantic emits boolean additionalProperties
try:
    import gradio_client.utils
    _orig_get_type = gradio_client.utils.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)
    gradio_client.utils.get_type = _safe_get_type

    _orig_json_schema = gradio_client.utils._json_schema_to_python_type
    def _safe_json_schema(schema, defs):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_json_schema(schema, defs)
    gradio_client.utils._json_schema_to_python_type = _safe_json_schema
except Exception:
    pass

try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is not None:
                return fn
            def wrapper(f):
                return f
            return wrapper
    spaces = _MockSpaces()

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

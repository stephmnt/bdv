from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_gradio_module():
    module_path = Path(__file__).resolve().parent / "app" / "gradio_app.py"
    spec = importlib.util.spec_from_file_location("gradio_app_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_gradio = _load_gradio_module()
demo = _gradio.create_interface()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

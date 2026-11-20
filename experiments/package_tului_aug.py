"""Repackage the trained all-data TULUI shuffle model with TULUI-fast inference."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import sys
import zipfile

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODEL = ROOT / "models" / "tului_aug_all"
FAST_SRC = ROOT / "models" / "tului_fast" / "your_model.py"
PACKAGE = ROOT / "submissions" / "candidates" / "TULUI_shuffle_aug.zip"


def main():
    graph = onnx.load(str(MODEL / "your_model.onnx"))
    metadata = {p.key: p.value for p in graph.metadata_props}
    config = json.loads(metadata["eh_config"])
    spec = importlib.util.spec_from_file_location("tului_aug_fast_package", FAST_SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    missing = sorted(set(config["features"]) - set(module.ALL_FEATURES))
    if missing:
        raise RuntimeError(f"Fast inference code lacks model features: {missing}")
    shutil.copyfile(FAST_SRC, MODEL / "your_model.py")
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(MODEL / "your_model.py", arcname="your_model.py")
        z.write(MODEL / "your_model.onnx", arcname="your_model.onnx")
    print(f"packaged {PACKAGE} ({PACKAGE.stat().st_size / 1e6:.2f} MB), "
          f"{len(config['features'])} features, trained_on={config.get('trained_on')}")


if __name__ == "__main__":
    main()

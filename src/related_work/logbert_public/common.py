from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def bootstrap_vendor_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    vendor_dir = base_dir / "vendor"
    os.environ.setdefault("MPLBACKEND", "Agg")
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    return vendor_dir


def _resolve_path(value: str | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    expanded = os.path.expanduser(str(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    root_dir = repo_root()
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML config requested, but PyYAML is not available in this environment."
                ) from exc
            data = yaml.safe_load(f)

    home_dir = _resolve_path(data.get("home_dir"), base_dir=root_dir)
    output_dir = _resolve_path(data["output_dir"], base_dir=root_dir)
    input_subdir = str(data.get("input_subdir", ".dataset/hdfs"))
    input_dir = (home_dir / input_subdir).resolve()

    normalized = dict(data)
    normalized["config_path"] = str(path)
    normalized["repo_root"] = str(repo_root())
    normalized["home_dir"] = str(home_dir)
    normalized["input_dir"] = str(input_dir)
    normalized["output_dir"] = str(output_dir)
    normalized["model_dir"] = str((output_dir / "bert").resolve())
    normalized["model_path"] = str((output_dir / "bert" / "best_bert.pth").resolve())
    normalized["train_vocab"] = str((output_dir / "train").resolve())
    normalized["vocab_path"] = str((output_dir / "vocab.pkl").resolve())
    normalized["scale_path"] = str((output_dir / "bert" / "scale.pkl").resolve())
    return normalized


def ensure_output_dirs(config: dict[str, Any]) -> None:
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["model_dir"]).mkdir(parents=True, exist_ok=True)


def build_options(config: dict[str, Any]) -> dict[str, Any]:
    options = dict(config["options"])

    force_cpu = bool(config.get("force_cpu", False))
    device = "cpu"
    with_cuda = False
    if not force_cpu:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            with_cuda = bool(options.get("with_cuda", True))

    options["device"] = device
    options["with_cuda"] = with_cuda
    options["output_dir"] = str(Path(config["output_dir"])) + "/"
    options["model_dir"] = str(Path(config["model_dir"])) + "/"
    options["model_path"] = config["model_path"]
    options["train_vocab"] = config["train_vocab"]
    options["vocab_path"] = config["vocab_path"]
    options["scale_path"] = config["scale_path"]
    return options

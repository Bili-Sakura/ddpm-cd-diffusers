"""
Logger and configuration utilities
====================================
Supports both YAML configs (recommended) and legacy JSON configs with
``//`` comments.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from typing import Any


def mkdirs(paths) -> None:
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


def _load_config(opt_path: str) -> dict:
    """Load a YAML or JSON config file."""
    ext = os.path.splitext(opt_path)[1].lower()
    with open(opt_path, "r") as f:
        if ext in (".yaml", ".yml"):
            import yaml  # PyYAML

            return yaml.safe_load(f)
        else:
            # Legacy JSON with // comments
            json_str = ""
            for line in f:
                json_str += line.split("//")[0] + "\n"
            return json.loads(json_str, object_pairs_hook=OrderedDict)


def parse(args) -> dict:
    """Parse CLI args + config file into a flat option dict."""
    phase = args.phase
    opt = _load_config(args.config)

    # Debug mode prefix
    if getattr(args, "debug", False):
        opt["name"] = "debug_{}".format(opt["name"])

    # Set up experiment root directory
    experiments_root = os.path.join(
        "experiments", "{}_{}".format(opt["name"], get_timestamp())
    )
    opt["path"]["experiments_root"] = experiments_root
    for key, path in opt["path"].items():
        if "resume" not in key and "experiments" not in key:
            opt["path"][key] = os.path.join(experiments_root, path)
            mkdirs(opt["path"][key])

    # Mirror CD paths if present
    if "path_cd" in opt:
        opt["path_cd"]["experiments_root"] = experiments_root
        for key, path in opt["path_cd"].items():
            if "resume" not in key and "experiments" not in key:
                opt["path_cd"][key] = os.path.join(experiments_root, path)
                mkdirs(opt["path_cd"][key])

    opt["phase"] = phase

    # GPU setup
    gpu_ids = getattr(args, "gpu_ids", None)
    if gpu_ids is not None:
        opt["gpu_ids"] = [int(x) for x in gpu_ids.split(",")]
    gpu_list = ",".join(str(x) for x in opt.get("gpu_ids", []))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    opt["distributed"] = len(opt.get("gpu_ids", [])) > 1

    # Debug overrides
    if "debug" in opt.get("name", ""):
        opt.setdefault("train", {})["val_freq"] = 2
        opt["train"]["print_freq"] = 2
        opt["train"]["save_checkpoint_freq"] = 3
        opt.setdefault("datasets", {}).setdefault("train", {})["batch_size"] = 2
        opt.setdefault("datasets", {}).setdefault("train", {})["data_len"] = 6
        opt.setdefault("datasets", {}).setdefault("val", {})["data_len"] = 3

    # W&B flag
    opt["enable_wandb"] = getattr(args, "enable_wandb", False)

    return opt


class NoneDict(dict):
    """A dict that returns ``None`` for missing keys."""

    def __missing__(self, key: str) -> None:
        return None


def dict_to_nonedict(opt: Any) -> Any:
    """Recursively convert dicts to :class:`NoneDict`."""
    if isinstance(opt, dict):
        return NoneDict(**{k: dict_to_nonedict(v) for k, v in opt.items()})
    elif isinstance(opt, list):
        return [dict_to_nonedict(v) for v in opt]
    return opt


def dict2str(opt: dict, indent_l: int = 1) -> str:
    """Format a nested dict as a human-readable string."""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


def setup_logger(
    logger_name,
    root: str,
    phase: str,
    level: int = logging.INFO,
    screen: bool = False,
) -> None:
    """Configure a named logger with a file handler."""
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(os.path.join(root, f"{phase}.log"), mode="w")
    fh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)

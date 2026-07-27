from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load(path: str = None) -> Dict[str, Any]:
    with open(path or _DEFAULT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override(cfg: Dict[str, Any], pairs: list) -> Dict[str, Any]:
    """Apply CLI overrides like ["audio.k_mad=3.5", "fuse.enter=0.4"]."""
    cfg = copy.deepcopy(cfg)
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError("unknown config key: %s" % key)
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg

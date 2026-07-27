"""轻量场景门控。

模块四的音画融合只能说明「有瞬态声音 + 有运动」，还不能说明画面是乒乓球。
这里先用不依赖模型的便宜特征做一道粗筛：固定机位乒乓球画面通常在中下区域
有球台、挡板、地胶等稳定长水平边缘；影视/朗读/生活素材往往弱很多。
"""
from __future__ import annotations

import json
import subprocess
from typing import Dict

import numpy as np


def assess(video_path: str, cfg: Dict) -> Dict:
    """返回视频级场景分数。score >= gate 视为通过。"""
    if not cfg.get("enabled", True):
        return {"score": 1.0, "edge": 1.0, "passed": True}

    meta = _probe(video_path)
    if meta is None:
        return {"score": 1.0, "edge": 1.0, "passed": True}

    width = int(cfg.get("width", 160))
    samples = int(cfg.get("samples", 5))
    gate = float(cfg.get("gate", 0.12))
    edges = []

    src_w, src_h, duration = meta
    height = max(1, int(round(src_h * width / float(src_w))))
    if duration <= 0:
        return {"score": 1.0, "edge": 1.0, "passed": True}

    start = max(0.5, duration * 0.1)
    end = max(start + 0.1, duration * 0.9)
    for t in np.linspace(start, end, samples):
        frame = _frame(video_path, float(t), width, height)
        if frame is None:
            continue
        edges.append(_horizontal_edge_strength(frame))

    edge = float(np.median(edges)) if edges else 1.0
    score = float(np.clip(edge / max(gate, 1e-6), 0, 1))
    return {"score": round(score, 3), "edge": round(edge, 3), "passed": edge >= gate}


def _probe(video_path: str):
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-select_streams", "v:0", "-show_streams", "-show_format", video_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return None
    info = json.loads(proc.stdout or b"{}")
    streams = info.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    duration = float(info.get("format", {}).get("duration", 0.0))
    return int(stream["width"]), int(stream["height"]), duration


def _frame(video_path: str, t: float, width: int, height: int):
    cmd = [
        "ffmpeg", "-v", "error", "-ss", "%.3f" % t, "-i", video_path,
        "-frames:v", "1", "-vf", "scale=%d:%d,format=rgb24" % (width, height),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = width * height * 3
    if proc.returncode != 0 or len(proc.stdout) < frame_size:
        return None
    return np.frombuffer(proc.stdout[:frame_size], dtype=np.uint8).reshape(height, width, 3)


def _horizontal_edge_strength(frame: np.ndarray) -> float:
    gray = (
        0.299 * frame[:, :, 0] +
        0.587 * frame[:, :, 1] +
        0.114 * frame[:, :, 2]
    ).astype(np.float32)
    vertical_gradient = np.abs(np.diff(gray, axis=0))
    h = vertical_gradient.shape[0]
    band = vertical_gradient[int(h * 0.25):int(h * 0.80)]
    if band.size == 0:
        return 0.0
    return float(np.percentile(band, 95) / 255.0)

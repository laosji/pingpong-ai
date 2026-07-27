"""画面运动强度。抽帧灰度帧差，够快也够用。

作用是给音频信号做交叉验证：光有击球声可能是隔壁球台传来的，
配合本画面确实有人在动，才判定为「本场在打」。
"""
from __future__ import annotations

import json
import subprocess
from typing import Dict, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # 允许在未装 opencv 时只跑音频
    cv2 = None


def motion_curve(video_path: str, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (时间戳, 运动强度 0-1)。"""
    if not cfg.get("enabled", True):
        return np.zeros(0), np.zeros(0)

    if cv2 is None or not _opencv_has_ffmpeg():
        return _motion_curve_ffmpeg(video_path, cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _motion_curve_ffmpeg(video_path, cfg)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / float(cfg["sample_fps"]))))
    width = int(cfg["width"])
    noise = int(cfg["pixel_noise"])
    roi = cfg.get("roi")

    times, vals = [], []
    prev = None
    idx = 0
    while True:
        ok = cap.grab()          # grab 不解码，跳帧很便宜
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                if roi:
                    x0, y0, rw, rh = roi
                    frame = frame[int(y0 * h):int((y0 + rh) * h),
                                  int(x0 * w):int((x0 + rw) * w)]
                    h, w = frame.shape[:2]
                small = cv2.resize(frame, (width, max(1, int(h * width / float(w)))))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    diff = cv2.absdiff(gray, prev)
                    vals.append(float((diff > noise).mean()))
                    times.append(idx / src_fps)
                prev = gray
        idx += 1
    cap.release()

    return _finish(times, vals)


def _motion_curve_ffmpeg(video_path: str, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """OpenCV 没有 FFmpeg 支持时，用 ffmpeg 抽灰度帧做同样的帧差。"""
    meta = _probe_video(video_path)
    if meta is None:
        return np.zeros(0), np.zeros(0)

    src_w, src_h = meta
    sample_fps = float(cfg["sample_fps"])
    width = int(cfg["width"])
    noise = int(cfg["pixel_noise"])
    roi = cfg.get("roi")

    crop_w, crop_h = src_w, src_h
    filters = []
    if roi:
        x0, y0, rw, rh = [float(x) for x in roi]
        crop_w = max(1, int(round(src_w * rw)))
        crop_h = max(1, int(round(src_h * rh)))
        crop_x = max(0, min(src_w - 1, int(round(src_w * x0))))
        crop_y = max(0, min(src_h - 1, int(round(src_h * y0))))
        crop_w = min(crop_w, src_w - crop_x)
        crop_h = min(crop_h, src_h - crop_y)
        filters.append("crop=%d:%d:%d:%d" % (crop_w, crop_h, crop_x, crop_y))

    height = max(1, int(round(crop_h * width / float(crop_w))))
    filters.extend(["fps=%.6f" % sample_fps, "scale=%d:%d" % (width, height), "format=gray"])

    cmd = [
        "ffmpeg", "-v", "error", "-i", video_path,
        "-an", "-vf", ",".join(filters),
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0), np.zeros(0)

    frame_size = width * height
    n_frames = len(proc.stdout) // frame_size
    if n_frames < 2:
        return np.zeros(0), np.zeros(0)

    raw = np.frombuffer(proc.stdout[:n_frames * frame_size], dtype=np.uint8)
    frames = raw.reshape(n_frames, height, width)
    diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    vals = (diffs > noise).mean(axis=(1, 2))
    times = np.arange(1, n_frames, dtype=np.float64) / sample_fps
    return _finish(times, vals)


def _probe_video(video_path: str):
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-select_streams", "v:0", "-show_streams", video_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return None
    info = json.loads(proc.stdout or b"{}")
    streams = info.get("streams") or []
    if not streams:
        return None
    s = streams[0]
    return int(s["width"]), int(s["height"])


def _opencv_has_ffmpeg() -> bool:
    if cv2 is None:
        return False
    try:
        return "FFMPEG:                      YES" in cv2.getBuildInformation()
    except Exception:
        return False


def _finish(times, vals) -> Tuple[np.ndarray, np.ndarray]:
    if len(vals) == 0:
        return np.zeros(0), np.zeros(0)

    v = np.array(vals, dtype=np.float32)
    # 用 95 分位归一化，避免个别镜头晃动把整条曲线压扁
    ref = np.percentile(v, 95)
    v = np.clip(v / ref, 0, 1) if ref > 1e-6 else np.zeros_like(v)
    return np.array(times, dtype=np.float64), v

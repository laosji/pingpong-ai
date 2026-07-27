"""画面运动强度。抽帧灰度帧差，够快也够用。

作用是给音频信号做交叉验证：光有击球声可能是隔壁球台传来的，
配合本画面确实有人在动，才判定为「本场在打」。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # 允许在未装 opencv 时只跑音频
    cv2 = None


def motion_curve(video_path: str, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (时间戳, 运动强度 0-1)。"""
    if cv2 is None or not cfg.get("enabled", True):
        return np.zeros(0), np.zeros(0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros(0), np.zeros(0)

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

    if not vals:
        return np.zeros(0), np.zeros(0)

    v = np.array(vals, dtype=np.float32)
    # 用 95 分位归一化，避免个别镜头晃动把整条曲线压扁
    ref = np.percentile(v, 95)
    v = np.clip(v / ref, 0, 1) if ref > 1e-6 else np.zeros_like(v)
    return np.array(times, dtype=np.float64), v

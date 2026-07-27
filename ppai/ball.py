"""球检测与追踪 —— 音频路线测到天花板 F1≈0.6 之后剩下的唯一一条路。

为什么不上 YOLO：没有任何标注框。为什么不上 TrackNet：为网球/羽毛球训的，
且乒乓球尺度差一个量级。固定机位下经典 CV 不需要训练数据，先拿它建立基线。

核心思路
--------
三帧差分：d = min(|f1-f0|, |f2-f1|)，只保留两步都在动的像素。
比背景建模更适合这里 —— 球速快，背景模型会把它当噪声吸收掉。

筛选靠尺度：球在 960×544 下约 6–10 像素（带运动模糊会拉成短条），
球员是几千像素的大块，按面积和外接框一刀切得很干净。

真正区分「球在飞」和「噪点」的是**连续三帧的匀速性**：
球在飞行段近似匀速直线（重力在 33ms 内的影响很小），
而噪点、反光、衣服褶皱不会连续三帧沿直线移动。
"""
from __future__ import annotations

import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def _read_bgr(video: str, t0: float, t1: float, width: int):
    """彩色帧。颜色是区分球和皮肤的关键 —— 实测白球饱和度 38-54，皮肤 102，
    灰度图里这两者亮度几乎一样（都在 150-190），根本分不开。"""
    meta = _probe(video)
    if meta is None:
        return np.zeros((0, 0, 0, 3), np.uint8), 30.0
    src_w, src_h, fps = meta
    height = max(1, int(round(src_h * width / float(src_w))))
    cmd = ["ffmpeg", "-v", "error", "-ss", "%.3f" % t0, "-i", video,
           "-t", "%.3f" % (t1 - t0), "-an",
           "-vf", "scale=%d:%d" % (width, height),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros((0, 0, 0, 3), np.uint8), fps
    n = len(proc.stdout) // (width * height * 3)
    return np.frombuffer(proc.stdout[:n * width * height * 3], np.uint8).reshape(
        n, height, width, 3), fps


def _read_gray(video: str, t0: float, t1: float, width: int) -> Tuple[np.ndarray, float, float]:
    """用 ffmpeg 抽指定时段的灰度帧。返回 (frames[N,H,W], fps, 实际宽高比缩放)。

    不降采样太狠 —— 球本来就只有几像素，缩一半就没了。
    """
    meta = _probe(video)
    if meta is None:
        return np.zeros((0, 0, 0), np.uint8), 30.0, 1.0
    src_w, src_h, fps = meta
    scale = width / float(src_w)
    height = max(1, int(round(src_h * scale)))

    cmd = ["ffmpeg", "-v", "error", "-ss", "%.3f" % t0, "-i", video,
           "-t", "%.3f" % (t1 - t0), "-an",
           "-vf", "scale=%d:%d,format=gray" % (width, height),
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros((0, 0, 0), np.uint8), fps, scale
    n = len(proc.stdout) // (width * height)
    frames = np.frombuffer(proc.stdout[:n * width * height], np.uint8)
    return frames.reshape(n, height, width), fps, scale


def _probe(video: str) -> Optional[Tuple[int, int, float]]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", video],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        w, h, rate = out.stdout.decode().split()
        num, _, den = rate.partition("/")
        return int(w), int(h), float(num) / float(den or 1)
    except Exception:
        return None


def candidates(frames: np.ndarray, cfg: Dict, hsv: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """逐帧找小运动目标。返回每帧的 [(x, y, area), ...]。"""
    if cv2 is None or len(frames) < 3:
        return [np.zeros((0, 4)) for _ in range(len(frames))]

    thr = int(cfg["diff_thr"])
    amin, amax = int(cfg["area_min"]), int(cfg["area_max"])
    bmax = int(cfg["box_max"])
    bmin = float(cfg["bright_min"])
    smax = float(cfg.get("sat_max", 70))
    vmin = float(cfg.get("val_min", 170))
    # ROI：把第二张球台、场边白色标识、球员白鞋这些又小又亮的干扰排除在外
    roi = cfg.get("roi")
    H, W = frames.shape[1], frames.shape[2]
    if roi:
        rx, ry, rw, rh = [float(z) for z in roi]
        x0, x1 = int(rx * W), int((rx + rw) * W)
        y0, y1 = int(ry * H), int((ry + rh) * H)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    out: List[np.ndarray] = [np.zeros((0, 4))]

    for i in range(1, len(frames) - 1):
        d1 = cv2.absdiff(frames[i], frames[i - 1])
        d2 = cv2.absdiff(frames[i + 1], frames[i])
        m = cv2.threshold(np.minimum(d1, d2), thr, 255, cv2.THRESH_BINARY)[1]
        # 开运算去掉单像素噪声，但核要小 —— 球本身就没几个像素
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if roi:
            mask = np.zeros_like(m); mask[y0:y1, x0:x1] = 255
            m = cv2.bitwise_and(m, mask)
        if hsv is not None:
            # 白球低饱和高明度；皮肤饱和度约为球的两倍，红地面更高
            cm = ((hsv[i][:, :, 1] < smax) & (hsv[i][:, :, 2] > vmin)).astype(np.uint8) * 255
            m = cv2.bitwise_and(m, cm)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        rows = []
        for k in range(1, n):
            x, y, w, h, a = stats[k]
            if a < amin or a > amax:
                continue
            if w > bmax or h > bmax:      # 球员、手臂这类大块
                continue
            # 面积完全区分不了球和噪点（实测中位数都是 6 像素），亮度才行：
            # 白球在蓝紫色场地上是最亮的小目标，这一条把候选从 77 个/帧压到 1.7 个/帧
            patch = frames[i][y:y + h, x:x + w]
            msk = (lab[y:y + h, x:x + w] == k)
            bright = float(patch[msk].mean()) if msk.any() else 0.0
            if bright < bmin:
                continue
            rows.append((cent[k][0], cent[k][1], a, bright))
        out.append(np.array(rows) if rows else np.zeros((0, 4)))
    out.append(np.zeros((0, 4)))
    return out


def flight_events(cand: List[np.ndarray], fps: float, cfg: Dict) -> np.ndarray:
    """找连续三帧近似匀速直线的候选点 —— 「球在飞」的证据。

    返回每帧的飞行证据数（0/1/2…）。噪点很难连续三帧沿直线走。
    """
    vmax = float(cfg["max_px_per_frame"])
    vmin = float(cfg["min_px_per_frame"])
    tol = float(cfg["accel_tol"])

    ev = np.zeros(len(cand))
    for i in range(1, len(cand) - 1):
        A, B, C = cand[i - 1], cand[i], cand[i + 1]
        if len(A) == 0 or len(B) == 0 or len(C) == 0:
            continue
        cnt = 0
        for b in B:
            va = A[:, :2] - b[:2]
            da = np.hypot(va[:, 0], va[:, 1])
            ia = np.flatnonzero((da >= vmin) & (da <= vmax))
            if len(ia) == 0:
                continue
            vc = C[:, :2] - b[:2]
            dc = np.hypot(vc[:, 0], vc[:, 1])
            ic = np.flatnonzero((dc >= vmin) & (dc <= vmax))
            if len(ic) == 0:
                continue
            # 匀速直线：前一步位移 ≈ -后一步位移
            for p in ia:
                pred = -va[p]
                err = np.hypot(vc[ic, 0] - pred[0], vc[ic, 1] - pred[1])
                if np.min(err) <= tol * max(da[p], 1.0):
                    cnt += 1
                    break
            if cnt:
                break      # 每帧最多算一次，避免多候选重复计数
        ev[i] = cnt
    return ev


def track(video: str, cfg: Dict, t0: float = 0.0,
          t1: Optional[float] = None) -> Dict[str, np.ndarray]:
    """返回 {times, flight, cand_count}。flight 是每帧的球飞行证据。"""
    dur = t1 if t1 is not None else 1e9
    frames, fps, _ = _read_gray(video, t0, dur, int(cfg["width"]))
    if len(frames) == 0:
        return {"times": np.zeros(0), "flight": np.zeros(0), "cand": np.zeros(0)}
    hsv = None
    if cfg.get("use_color", True):
        col, _ = _read_bgr(video, t0, dur, int(cfg["width"]))
        if len(col) == len(frames):
            hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in col])
    cand = candidates(frames, cfg, hsv)
    ev = flight_events(cand, fps, cfg)
    times = t0 + np.arange(len(frames)) / fps
    return {"times": times, "flight": ev,
            "cand": np.array([len(c) for c in cand], dtype=float)}


def rally_score(res: Dict[str, np.ndarray], dt: float, grid: np.ndarray,
                win_s: float = 1.0) -> np.ndarray:
    """把逐帧飞行证据聚合成「正在打球」的分数。

    对打时球连续往返，飞行证据密集；准备发球时球在手上或小幅弹跳，
    形不成持续的高速直线轨迹。
    """
    t, ev = res["times"], res["flight"]
    if len(t) == 0:
        return np.zeros(len(grid))
    half = win_s / 2.0
    left = np.searchsorted(t, grid - half)
    right = np.searchsorted(t, grid + half)
    counts = np.array([ev[a:b].sum() for a, b in zip(left, right)])
    # 1 秒窗内 30 帧，对打时期望有相当比例的帧能看到球在飞
    return np.clip(counts / (win_s * 12.0), 0, 1)

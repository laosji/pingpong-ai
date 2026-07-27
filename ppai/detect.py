"""信号融合 + 状态机，输出有效比赛片段。对应方案「模块四」。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def hit_density(hits: np.ndarray, grid: np.ndarray, half_win_s: float) -> np.ndarray:
    """每个时刻前后 half_win 秒内的击球次数，再归一化。"""
    if len(hits) == 0:
        return np.zeros(len(grid))
    left = np.searchsorted(hits, grid - half_win_s, side="left")
    right = np.searchsorted(hits, grid + half_win_s, side="right")
    counts = (right - left).astype(np.float64)
    # 对打时 2 秒窗内通常 4 拍以上；到 6 拍即视为满分
    return np.clip(counts / 6.0, 0, 1)


def fuse(hits: np.ndarray, m_times: np.ndarray, m_vals: np.ndarray,
         duration: float, cfg: Dict) -> Dict[str, np.ndarray]:
    dt = cfg["dt"]
    grid = np.arange(0.0, max(duration, dt), dt)

    a = hit_density(hits, grid, cfg["hit_window_s"])
    if len(m_times) > 1:
        m = np.interp(grid, m_times, m_vals, left=m_vals[0], right=m_vals[-1])
        w_a, w_m = cfg["w_audio"], cfg["w_motion"]
    else:
        m = np.zeros(len(grid))
        w_a, w_m = 1.0, 0.0   # 没有运动信号就全靠音频

    score = w_a * a + w_m * m
    return {"grid": grid, "score": score, "audio": a, "motion": m}


def segment(grid: np.ndarray, score: np.ndarray, duration: float,
            hits: np.ndarray, cfg: Dict) -> List[Dict]:
    """滞回状态机：IDLE -> READY -> PLAYING -> END。"""
    dt = cfg["dt"]
    enter, exit_ = cfg["enter"], cfg["exit"]
    need_enter = int(round(cfg["min_enter_s"] / dt))
    need_exit = int(round(cfg["end_silence_s"] / dt))

    raw = []
    state = "IDLE"
    above = 0
    below = 0
    start_i = 0

    for i, s in enumerate(score):
        if state == "IDLE":
            if s >= enter:
                above += 1
                if above >= need_enter:
                    state = "PLAYING"
                    start_i = i - above + 1     # 回溯到起振点
                    below = 0
            else:
                above = 0
        else:  # PLAYING
            if s < exit_:
                below += 1
                if below >= need_exit:
                    raw.append((start_i, i - below + 1))
                    state = "IDLE"
                    above = 0
                    below = 0
            else:
                below = 0

    if state == "PLAYING":
        raw.append((start_i, len(score) - 1))

    # 合并 -> 补边 -> 过滤
    merged = []
    for a, b in raw:
        if merged and (a - merged[-1][1]) * dt <= cfg["merge_gap_s"]:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    out = []
    for a, b in merged:
        start = max(0.0, a * dt - cfg["pad_start_s"])
        end = min(duration, b * dt + cfg["pad_end_s"])
        if end - start < cfg["min_segment_s"]:
            continue
        n_hits = int(np.sum((hits >= start) & (hits <= end)))
        conf = float(np.clip(score[a:max(b, a + 1)].mean() / max(enter, 1e-6), 0, 1))
        out.append({
            "id": len(out) + 1,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "hit_count": n_hits,
            "confidence": round(conf, 3),
        })
    return out

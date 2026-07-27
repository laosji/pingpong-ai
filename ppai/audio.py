"""击球声检测。

思路：乒乓球击球是一个极短的宽带瞬态（~5-15ms），能量集中在 1-5kHz，
起振非常陡。人声、球馆嗡鸣、空调这类干扰要么频率低、要么起振缓慢，
所以在限定频带上做「谱通量（spectral flux）」+ 自适应阈值就能把它挑出来，
不需要神经网络。这是第一版；升级路线是 YAMNet / PANNs。
"""
from __future__ import annotations

import subprocess
from typing import Dict, Tuple

import numpy as np


def extract_pcm(video_path: str, sr: int) -> np.ndarray:
    """用 ffmpeg 抽单声道 float32 PCM。无音轨时返回空数组。"""
    cmd = [
        "ffmpeg", "-v", "error", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def onset_envelope(x: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, float]:
    """返回 (谱通量包络, 帧率)。分块处理，一小时视频也只占几 MB 内存。"""
    sr, n_fft, hop = cfg["sr"], cfg["n_fft"], cfg["hop"]
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= cfg["fmin"]) & (freqs <= cfg["fmax"])

    frame_rate = sr / float(hop)
    if len(x) < n_fft:
        return np.zeros(0, dtype=np.float32), frame_rate
    n_frames = 1 + (len(x) - n_fft) // hop

    env = np.zeros(n_frames, dtype=np.float32)
    offsets = np.arange(n_fft)
    prev_tail = None
    block = 4000

    for s in range(0, n_frames, block):
        e = min(s + block, n_frames)
        idx = (np.arange(s, e) * hop)[:, None] + offsets[None, :]
        frames = x[idx] * win
        mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)[:, band]
        # 对数压缩：让检测对整体音量不敏感，只看相对突变
        logmag = np.log1p(mag * 100.0)
        head = prev_tail if prev_tail is not None else logmag[0]
        shifted = np.vstack([head[None, :], logmag[:-1]])
        # 只取正差分 —— 我们要的是「突然出现的能量」，不是衰减
        env[s:e] = np.maximum(logmag - shifted, 0.0).sum(axis=1)
        prev_tail = logmag[-1]

    return env, frame_rate


def _adaptive_threshold(env: np.ndarray, frame_rate: float, win_s: float, k: float) -> np.ndarray:
    """局部中位数 + k*MAD。球馆底噪基本平稳，用鲁棒统计比全局阈值稳得多。

    按整块算统计量再线性插值，避免 O(N*W) 的滑动中位数。
    """
    w = max(1, int(round(win_s * frame_rate)))
    n_blocks = max(1, int(np.ceil(len(env) / w)))
    pad = n_blocks * w - len(env)
    padded = np.concatenate([env, np.repeat(env[-1:], pad)]) if pad else env
    blocks = padded.reshape(n_blocks, w)

    med = np.median(blocks, axis=1)
    mad = np.median(np.abs(blocks - med[:, None]), axis=1) * 1.4826

    centers = np.arange(n_blocks) * w + w / 2.0
    grid = np.arange(len(env), dtype=np.float64)
    med_i = np.interp(grid, centers, med)
    mad_i = np.interp(grid, centers, mad)
    # MAD 可能为 0（极安静段），给个下限防止阈值塌到 0 触发满屏误检
    floor = np.percentile(env, 60) * 0.05 if len(env) else 0.0
    return med_i + k * np.maximum(mad_i, floor)


def detect_hits(x: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """返回 (击球时间点, 包络, 阈值曲线, 帧率)。"""
    env, frame_rate = onset_envelope(x, cfg)
    if len(env) == 0:
        return np.zeros(0), env, np.zeros(0), frame_rate

    thr = _adaptive_threshold(env, frame_rate, cfg["noise_win_s"], cfg["k_mad"])

    # 局部极大值且超过阈值
    is_peak = np.zeros(len(env), dtype=bool)
    is_peak[1:-1] = (env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > thr[1:-1])
    cand = np.flatnonzero(is_peak)
    if len(cand) == 0:
        return np.zeros(0), env, thr, frame_rate

    # 按强度降序贪心去重，保证最小间隔
    min_gap = max(1, int(round(cfg["min_gap_s"] * frame_rate)))
    order = cand[np.argsort(-env[cand])]
    kept = []
    taken = np.zeros(len(env), dtype=bool)
    for i in order:
        lo, hi = max(0, i - min_gap), min(len(env), i + min_gap + 1)
        if not taken[lo:hi].any():
            kept.append(i)
            taken[i] = True

    hits = np.sort(np.array(kept)) / frame_rate
    return hits, env, thr, frame_rate

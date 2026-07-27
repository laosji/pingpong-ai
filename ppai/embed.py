"""PANNs CNN14 音频嵌入（方案「模块四」的第二条路线）。

为什么不直接用 AudioSet 的类别输出：AudioSet 里没有「乒乓球击球」这一类，
最接近的 Ping / Tick / Slap 都不对。正确用法是把预训练网络当**特征提取器**，
在自己标的数据上训一个小分类器 —— 这也是标注数据（模块九）的直接用途。

权重放在 ~/panns_data/Cnn14_mAP=0.431.pth（312MB），首次需手动下载：
  curl -L -o ~/panns_data/Cnn14_mAP=0.431.pth \
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
  curl -L -o ~/panns_data/class_labels_indices.csv \
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

PANNS_SR = 32000
CKPT = os.path.expanduser("~/panns_data/Cnn14_mAP=0.431.pth")

_model = None


def available() -> bool:
    return os.path.exists(CKPT)


def _load():
    global _model
    if _model is not None:
        return _model
    import torch
    from panns_inference.models import Cnn14

    if not available():
        raise FileNotFoundError("缺少 PANNs 权重: %s（见 embed.py 顶部下载命令）" % CKPT)
    m = Cnn14(sample_rate=PANNS_SR, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    _model = m
    return m


def embed_windows(pcm32k: np.ndarray, win_s: float = 1.0, hop_s: float = 0.5,
                  batch: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """滑窗提嵌入。返回 (窗中心时间, [N, 2048])。"""
    import torch

    m = _load()
    win = int(win_s * PANNS_SR)
    hop = int(hop_s * PANNS_SR)
    if len(pcm32k) < win:
        pcm32k = np.pad(pcm32k, (0, win - len(pcm32k)))
    starts = np.arange(0, len(pcm32k) - win + 1, hop)
    times = (starts + win / 2.0) / PANNS_SR

    feats = []
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunk = np.stack([pcm32k[s:s + win] for s in starts[i:i + batch]])
            out = m(torch.from_numpy(chunk).float())
            feats.append(out["embedding"].numpy())
    return times, np.concatenate(feats) if feats else np.zeros((0, 2048), dtype=np.float32)


def embed_video(video: str, win_s: float = 1.0, hop_s: float = 0.5,
                cache_dir: Optional[str] = "cache") -> Tuple[np.ndarray, np.ndarray]:
    """抽音轨 -> 嵌入。带磁盘缓存 —— 提特征比后续训练慢得多，反复调参时省时间。"""
    from . import audio as A

    key = None
    if cache_dir:
        stem = os.path.splitext(os.path.basename(video))[0]
        key = os.path.join(cache_dir, "%s_%gs_%gs.npz" % (stem, win_s, hop_s))
        if os.path.exists(key):
            d = np.load(key)
            return d["times"], d["feats"]

    pcm = A.extract_pcm(video, PANNS_SR)
    times, feats = embed_windows(pcm, win_s, hop_s)
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(key, times=times, feats=feats)
    return times, feats

"""PANNs CNN14 音频嵌入（方案「模块四」的第二条路线）。

为什么不直接用 AudioSet 的类别输出：AudioSet 里没有「乒乓球击球」这一类，
最接近的 Ping / Tick / Slap 都不对。正确用法是把预训练网络当**特征提取器**，
在自己标的数据上训一个小分类器 —— 这也是标注数据（模块九）的直接用途。

两套后端，优先 ONNX
--------------------
桌面 app 要装到用户机器上，torch 一个库就 529MB，占整包一半还多。
CNN14 导出成 ONNX 后只剩 323MB 权重 + 75MB onnxruntime，而且**数值等价**
（实测 7.4 分钟素材 886 个窗，最大绝对差 2.5e-05，重排概率最大差 1.1e-05，
按比例保留 60% 时选中的候选完全相同，速度 0.98x）。

所以运行时走 ONNX，torch 只在导出和训练时用：

    .venv/bin/python -m ppai.cli export-onnx     # 生成 models/cnn14.onnx

torch 权重放在 ~/panns_data/Cnn14_mAP=0.431.pth（312MB），首次需手动下载：
  curl -L -o ~/panns_data/Cnn14_mAP=0.431.pth \
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

PANNS_SR = 32000
CKPT = os.path.expanduser("~/panns_data/Cnn14_mAP=0.431.pth")
from .paths import DATA, MODELS


def _onnx_path() -> str:
    """查找顺序：环境变量 → 数据目录（按需下载的落点）→ 应用包内。

    **每次调用都重新解析，不缓存成模块常量。** 模型是运行时下载的，
    进程启动时可能还不在；缓存成常量的话，下载完还得重启 app 才生效。
    """
    env = os.environ.get("PIPO_CNN14_ONNX")
    if env:
        return env
    for p in (os.path.join(DATA, "models", "cnn14.onnx"),
              os.path.join(MODELS, "cnn14.onnx")):
        if os.path.exists(p):
            return p
    return os.path.join(DATA, "models", "cnn14.onnx")


ONNX_PATH = _onnx_path()

_model = None      # torch
_sess = None       # onnxruntime


def available() -> bool:
    """能不能提嵌入 —— 两个后端有一个就行。"""
    return os.path.exists(_onnx_path()) or os.path.exists(CKPT)


def _load():
    """torch 版。只在导出 ONNX 和重新训练时用，app 里不会走到。"""
    global _model
    if _model is not None:
        return _model
    import torch
    from panns_inference.models import Cnn14

    if not os.path.exists(CKPT):
        raise FileNotFoundError("缺少 PANNs 权重: %s（见 embed.py 顶部下载命令）" % CKPT)
    m = Cnn14(sample_rate=PANNS_SR, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    _model = m
    return m


def _load_onnx():
    global _sess
    if _sess is not None:
        return _sess
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    _sess = ort.InferenceSession(_onnx_path(), so, providers=["CPUExecutionProvider"])
    return _sess


def export_onnx(dst: str = ONNX_PATH) -> str:
    """把 torch 版 CNN14 导成 ONNX。构建 app 时跑一次，用户机器上不需要。

    只导 embedding 这一路输出 —— 527 类的分类头我们从来不用（AudioSet 里
    没有「乒乓球击球」这一类），不导能省一点，也让接口更明确。
    """
    import torch

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)["embedding"]

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.onnx.export(
        _Wrap(_load()).eval(), (torch.zeros(4, PANNS_SR),), dst,
        input_names=["pcm"], output_names=["embedding"],
        # batch 必须是动态轴：最后一批通常不满 64
        dynamic_axes={"pcm": {0: "n"}, "embedding": {0: "n"}},
        opset_version=17, dynamo=False)
    return dst


def embed_windows(pcm32k: np.ndarray, win_s: float = 1.0, hop_s: float = 0.5,
                  batch: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """滑窗提嵌入。返回 (窗中心时间, [N, 2048])。有 ONNX 就走 ONNX。"""
    win = int(win_s * PANNS_SR)
    hop = int(hop_s * PANNS_SR)
    if len(pcm32k) < win:
        pcm32k = np.pad(pcm32k, (0, win - len(pcm32k)))
    starts = np.arange(0, len(pcm32k) - win + 1, hop)
    times = (starts + win / 2.0) / PANNS_SR

    feats = []
    if os.path.exists(_onnx_path()):
        sess = _load_onnx()
        for i in range(0, len(starts), batch):
            chunk = np.stack([pcm32k[s:s + win] for s in starts[i:i + batch]])
            feats.append(sess.run(["embedding"], {"pcm": chunk.astype(np.float32)})[0])
    else:
        import torch
        m = _load()
        with torch.no_grad():
            for i in range(0, len(starts), batch):
                chunk = np.stack([pcm32k[s:s + win] for s in starts[i:i + batch]])
                feats.append(m(torch.from_numpy(chunk).float())["embedding"].numpy())
    return times, np.concatenate(feats) if feats else np.zeros((0, 2048), dtype=np.float32)


def embed_video(video: str, win_s: float = 1.0, hop_s: float = 0.5,
                cache_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """抽音轨 -> 嵌入。带磁盘缓存 —— 提特征比后续训练慢得多，反复调参时省时间。"""
    from . import audio as A
    from .paths import CACHE

    if cache_dir is None:
        cache_dir = CACHE
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

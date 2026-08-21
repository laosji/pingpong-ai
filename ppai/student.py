"""蒸馏出来的小模型：替掉 323MB 的 CNN14。

为什么这件事做得成
------------------
整条下游其实只是**2048 维嵌入上的一个逻辑回归**（rerank.bin 才 8KB）。
我们从来不需要那 2048 维，只需要最后那个标量概率。所以要蒸的不是
「一个通用音频编码器」，而是「一个乒乓球击球声概率函数」——
后者的信息量小得多。

为什么之前几次换小骨干都失败了，而这次不一样
--------------------------------------------
docs/RESEARCH.md 里记着三次失败（高分辨率 log-mel、宽窗 log-mel、
两个 MobileNetV3），共同点是**拿 8 份标注从头训**，只有 3434 个候选，
数据饿死。蒸馏不需要标注：任意音频喂给教师就得到一个连续目标，
增广之后样本量没有上限。学的是教师这个**函数**，不是那几段素材。

设计约束
--------
**输入签名必须和 CNN14 一模一样**（B, 32000 的 float32 波形）。
mel 前端做进图里而不是放到 Kotlin ——
  * Kotlin 侧只需把「拿 2048 维再算逻辑回归」换成「直接读输出」
  * 不引入新的跨语言一致性面（前端一旦分裂，两边就会慢慢走散）
代价是 DFT 基当成固定权重存进模型（约 1MB），换来的是零移植风险。

STFT 用 conv1d + 固定 DFT 基，不用 torch.stft ——
后者导出成 ONNX 的 STFT 算子，在 Android 的 onnxruntime 上支持情况
没验证过，为了省 1MB 冒这个险不值得。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

PANNS_SR = 32000
WIN_SAMPLES = 32000        # 1 秒，和教师的窗一致
N_FFT = 512
HOP = 160                  # 32k / 160 = 200 帧每秒
N_MELS = 64
FMIN, FMAX = 50.0, 14000.0


def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float, fmax: float) -> np.ndarray:
    """标准 Slaney 风格三角滤波器组。自己写而不是引 librosa ——
    这一段要跟着模型走，少一个依赖少一处版本漂移。"""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_bins)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, ctr, hi = pts[i], pts[i + 1], pts[i + 2]
        left = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(left, right))
    return fb


class LogMel(nn.Module):
    """波形 -> log-mel。全部是固定权重，不参与训练。"""

    def __init__(self):
        super().__init__()
        n_bins = N_FFT // 2 + 1
        win = np.hanning(N_FFT).astype(np.float32)
        k = np.arange(N_FFT)
        # 实部和虚部各一组卷积核，拼成一个 conv1d
        ang = -2.0 * np.pi * np.outer(np.arange(n_bins), k) / N_FFT
        basis = np.concatenate([np.cos(ang), np.sin(ang)]).astype(np.float32) * win
        self.register_buffer("dft", torch.from_numpy(basis).unsqueeze(1))
        self.register_buffer(
            "mel", torch.from_numpy(_mel_filterbank(PANNS_SR, N_FFT, N_MELS, FMIN, FMAX)))
        self.n_bins = n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 32000) -> (B, 1, 32000)
        z = torch.nn.functional.conv1d(x.unsqueeze(1), self.dft, stride=HOP)
        re, im = z[:, :self.n_bins], z[:, self.n_bins:]
        power = re * re + im * im                       # (B, n_bins, T)
        m = torch.matmul(self.mel, power)               # (B, n_mels, T)
        # log1p 而不是 log：安静段的功率会到 0，log 会变 -inf
        return torch.log1p(m * 1e4)


class Student(nn.Module):
    """log-mel + 小卷积网 -> 一个 logit。

    输出**故意是 logit 而不是概率**：Kotlin 侧原来就是拿 2048 维算一个
    logit 再 sigmoid，保持同一层语义，阈值和门禁的数值口径都不用改。
    """

    def __init__(self, width: int = 1):
        super().__init__()
        self.front = LogMel()
        c = [max(8, int(w * width)) for w in (24, 48, 96, 128)]

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.body = nn.Sequential(
            blk(1, c[0]), blk(c[0], c[1]), blk(c[1], c[2]), blk(c[2], c[3]))
        self.head = nn.Linear(c[3], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.front(x).unsqueeze(1)          # (B, 1, mel, T)
        h = self.body(h)
        h = h.mean(dim=(2, 3))                  # 全局平均池化
        return self.head(h).squeeze(-1)         # (B,) logit


def n_params(m: nn.Module) -> int:
    """只数会训练的参数。DFT 基和 mel 滤波器是常量，占体积但不算容量。"""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

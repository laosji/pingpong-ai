"""时间轴可视化 —— 调参时肉眼校准用，这是原型阶段最值钱的一张图。"""
from __future__ import annotations

from typing import Dict, List

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 文件名里全是中文，标题渲染不出来图就白画了
for _p in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
           "/System/Library/Fonts/Hiragino Sans GB.ttc"):
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_p).get_name()
        matplotlib.rcParams["axes.unicode_minus"] = False
        break


def plot(out_png: str, title: str, env: np.ndarray, thr: np.ndarray, frame_rate: float,
         hits: np.ndarray, fused: Dict[str, np.ndarray], segments: List[Dict],
         cfg: Dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(title, fontsize=11)

    ax = axes[0]
    if len(env):
        t = np.arange(len(env)) / frame_rate
        ax.plot(t, env, lw=0.4, color="#4C72B0", label="spectral flux")
        ax.plot(t, thr, lw=0.8, color="#C44E52", label="adaptive threshold")
    ax.vlines(hits, 0, ax.get_ylim()[1], color="#DD8452", lw=0.5, alpha=0.7,
              label="detected hits (%d)" % len(hits))
    ax.set_ylabel("audio onset")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    ax.plot(fused["grid"], fused["audio"], lw=0.8, color="#DD8452", label="hit density")
    ax.plot(fused["grid"], fused["motion"], lw=0.8, color="#55A868", label="motion")
    ax.set_ylabel("signals")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[2]
    ax.plot(fused["grid"], fused["score"], lw=1.0, color="#333333", label="fused score")
    ax.axhline(cfg["fuse"]["enter"], color="#C44E52", ls="--", lw=0.7, label="enter")
    ax.axhline(cfg["fuse"]["exit"], color="#8172B3", ls="--", lw=0.7, label="exit")
    for s in segments:
        ax.axvspan(s["start"], s["end"], color="#55A868", alpha=0.25)
        ax.text(s["start"], 1.02, "#%d" % s["id"], fontsize=7, color="#2A6F3F")
    ax.set_ylabel("score")
    ax.set_xlabel("time (s)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

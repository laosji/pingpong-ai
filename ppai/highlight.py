"""精彩集锦（方案模块六「精彩评分」+ 模块七「自动剪辑」）。

为什么这个能用而「剪掉所有空闲」不能
------------------------------------
两者对指标的要求不同：
  * 剪空闲要高召回 —— 漏掉任何一个回合都是缺陷。实测 F1 只有 0.618，不够。
  * 集锦只要 top-K 精准 —— 从一小时里挑 10 个最精彩的，漏掉第 11 个无所谓。

实测（12 分钟录像，按瞬态数排序）：Top-10 命中率 100%，Top-20 为 90%。
排序信号本身是可靠的，即使逐帧判定并不可靠。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np


def rallies(hits: np.ndarray, cfg: Dict) -> List[Dict]:
    """把击球瞬态按间隔聚成回合。"""
    gap = float(cfg["gap_s"])
    if len(hits) == 0:
        return []
    out = [{"start": float(hits[0]), "end": float(hits[0]), "hits": 1}]
    for t in hits[1:]:
        if t - out[-1]["end"] <= gap:
            out[-1]["end"] = float(t)
            out[-1]["hits"] += 1
        else:
            out.append({"start": float(t), "end": float(t), "hits": 1})
    return [r for r in out
            if r["end"] - r["start"] >= cfg["min_duration_s"] and r["hits"] >= cfg["min_hits"]]


def score(rs: List[Dict], m_times: np.ndarray, m_vals: np.ndarray, cfg: Dict) -> List[Dict]:
    """方案模块六的评分，按能实测的量重新分配权重。

    原方案：相持30% + 速度20% + 移动范围20% + 动作幅度20% + 结果10%。
    「结果」（得分/失误）现在判不了，去掉；「速度」和「动作幅度」都只能
    由击球密度和画面运动近似，合并。权重放在 config 里，方便后续调整。
    """
    if not rs:
        return []
    w = cfg["weights"]
    n_hits = np.array([r["hits"] for r in rs], float)
    dur = np.array([r["end"] - r["start"] for r in rs], float)
    rate = n_hits / np.maximum(dur, 1e-6)
    if len(m_times) > 1:
        motion = np.array([m_vals[(m_times >= r["start"]) & (m_times <= r["end"])].mean()
                           if np.any((m_times >= r["start"]) & (m_times <= r["end"])) else 0.0
                           for r in rs])
    else:
        motion = np.zeros(len(rs))

    def nz(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    s = (w["rally_length"] * nz(n_hits) + w["duration"] * nz(dur)
         + w["hit_rate"] * nz(rate) + w["motion"] * nz(motion))
    for r, v, m in zip(rs, s, motion):
        r["score"] = round(float(v), 4)
        r["motion"] = round(float(m), 3)
        r["hit_rate"] = round(float(r["hits"] / max(r["end"] - r["start"], 1e-6)), 2)
    return sorted(rs, key=lambda r: -r["score"])


def select(rs: List[Dict], cfg: Dict, total_s: Optional[float] = None) -> List[Dict]:
    """取前 N 个（或凑满目标时长），再按时间顺序排列 —— 集锦按时间线看更自然。"""
    pad_a, pad_b = cfg["pad_start_s"], cfg["pad_end_s"]
    picked, acc = [], 0.0
    for r in rs:
        if total_s is not None and acc >= total_s:
            break
        if total_s is None and len(picked) >= cfg["top_n"]:
            break
        picked.append(r)
        acc += r["end"] - r["start"] + pad_a + pad_b
    # 扩边后相邻片段会重叠，直接输出会在成片里出现重复画面。
    # 重叠说明它们本就是被 gap 规则切开的同一段相持，合并回去。
    merged: List[Dict] = []
    for r in sorted(picked, key=lambda x: x["start"]):
        a, b = max(0.0, r["start"] - pad_a), r["end"] + pad_b
        if merged and a <= merged[-1]["_b"]:
            m = merged[-1]
            m["_b"] = max(m["_b"], b)
            m["hits"] += r["hits"]
            m["score"] = max(m["score"], r["score"])
        else:
            merged.append({"_a": a, "_b": b, "hits": r["hits"], "score": r["score"]})

    out = []
    for i, m in enumerate(merged, 1):
        dur = m["_b"] - m["_a"]
        out.append({
            "id": i,
            "start": round(m["_a"], 2),
            "end": round(m["_b"], 2),
            "duration": round(dur, 2),
            "hit_count": m["hits"],
            "hit_rate": round(m["hits"] / max(dur, 1e-6), 2),
            "confidence": round(m["score"], 4),
        })
    return out

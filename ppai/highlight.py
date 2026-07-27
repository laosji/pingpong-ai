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


def rallies(hits: np.ndarray, cfg: Dict, amps: Optional[np.ndarray] = None) -> List[Dict]:
    """把击球瞬态按间隔聚成**活动片段**。

    注意用词：这里聚出来的不是「回合」。对着穷尽标注实测，
    这段训练录像里真实回合只有 0.5-2.8 秒、2-5 拍，120 秒里 14 个，
    中间隔着 8-10 秒的捡球准备。而本函数输出的片段是 10-40 秒 ——
    它是若干短回合加弹跳、噪声连成的活动密集区。

    对训练录像的集锦来说这反而合适（连续练球比孤立的两板更好看），
    但 hit_count 不能理解成「这个回合打了多少拍」。
    """
    gap = float(cfg["gap_s"])
    if len(hits) == 0:
        return []
    if amps is None:
        amps = np.ones(len(hits))
    idx: List[List[int]] = [[0]]
    for i in range(1, len(hits)):
        if hits[i] - hits[idx[-1][-1]] <= gap:
            idx[-1].append(i)
        else:
            idx.append([i])

    out = []
    for g in idx:
        ts = np.asarray(hits)[g].astype(float)
        av = np.asarray(amps)[g].astype(float)
        # 落地弹跳会让间隔一直很密，2.5 秒的分段规则不会在那里断开，
        # 回合于是被拖长，把捡球画面卷进来。在弹跳起点截断。
        bounce_end = False
        if cfg.get("trim_bounce", True):
            # 1) 回合内部混进了弹跳 —— 截断
            b = find_bounce_decay(ts, cfg)
            if b is not None and b > ts[0]:
                keep = ts <= b
                ts, av = ts[keep], av[keep]
                bounce_end = True
            else:
                # 2) gap 降到 0.7 之后弹跳会被切成**独立的簇**，不再落在回合内部。
                #    所以要往回合结束之后看：紧随其后的瞬态是不是弹跳衰减。
                tail_win = float(cfg.get("bounce_lookahead_s", 3.0))
                after = np.asarray(hits)
                after = after[(after > ts[-1]) & (after <= ts[-1] + tail_win)]
                if len(after) >= cfg["bounce_min_count"]:
                    if find_bounce_decay(np.concatenate([ts[-1:], after]), cfg) is not None:
                        bounce_end = True
        if len(ts) == 0:
            continue
        # 力量：取较强的那部分击球，而不是均值 —— 一个回合里总有轻挡和过渡球，
        # 用均值会把爆发力强的回合和平稳的回合拉平
        out.append({
            "start": float(ts[0]), "end": float(ts[-1]), "hits": len(ts),
            "power": float(np.percentile(av, 80)) if len(av) else 0.0,
            # 绝杀用：最后两拍的力量。回合以一记重杀结束才算绝杀
            "tail_power": float(av[-2:].max()) if len(av) else 0.0,
            # 失误/结束用：这个回合是不是以「球落地连续弹跳」收尾
            "ended_with_bounce": bounce_end,
        })
    return [r for r in out
            if r["end"] - r["start"] >= cfg["min_duration_s"] and r["hits"] >= cfg["min_hits"]]


def find_bounce_decay(ts: np.ndarray, cfg: Dict) -> Optional[float]:
    """找球落地后的连续弹跳，返回弹跳起点时刻（即回合真正结束的地方）。

    物理依据：弹跳的恢复系数是常数，每次弹跳保留固定比例的能量，
    所以相邻间隔按**固定比值**收缩。实测一例 106.53-108.48s：
        间隔 0.74 0.55 0.32 0.21 0.13，比值 0.74 0.58 0.66 0.62（标准差 0.06）
    而回合内的击球虽然也可能出现间隔递减，比值却是散的：
        104.96s 处 间隔 0.61 0.48 0.16 0.10，比值 0.79 0.33 0.63（标准差 0.19）
    只看「递减」会把真实回合切断，必须加比值一致性。

    不加这个判据的后果：回合被弹跳声拖长，捡球画面被剪进集锦
    （实测集锦 #1 的 8-12 秒就是捡球）。
    """
    if len(ts) < cfg["bounce_min_count"] + 1:
        return None
    ioi = np.diff(ts)
    best = None
    for i in range(len(ioi) - cfg["bounce_min_count"] + 1):
        j = i
        while j + 1 < len(ioi) and ioi[j + 1] < ioi[j]:
            j += 1
        n = j - i + 1
        if n < cfg["bounce_min_count"]:
            continue
        seg = ioi[i:j + 1]
        if seg[-1] > cfg["bounce_final_ioi"]:      # 弹跳末尾必然很密
            continue
        ratios = seg[1:] / np.maximum(seg[:-1], 1e-6)
        if ratios.std() > cfg["bounce_ratio_std"]:  # 比值必须一致
            continue
        if not (cfg["bounce_ratio_lo"] <= ratios.mean() <= cfg["bounce_ratio_hi"]):
            continue
        if best is None or ts[i] < best:
            best = float(ts[i])
    return best


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
    power = np.array([r.get("power", 0.0) for r in rs], float)
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
         + w["hit_rate"] * nz(rate) + w["motion"] * nz(motion)
         + w.get("power", 0.0) * nz(power))
    for r, v, m in zip(rs, s, motion):
        r["score"] = round(float(v), 4)
        r["motion"] = round(float(m), 3)
        r["power"] = round(float(r.get("power", 0.0)), 1)
        r["hit_rate"] = round(float(r["hits"] / max(r["end"] - r["start"], 1e-6)), 2)
    return rs


# 「绝杀」和「失误」是同一个事件的两面：一方的绝杀就是另一方的失分，
# 一个回合只有一次「最后一拍」。所以不做成两个独立类别，而是**同一根轴的两端**：
#     收尾力量 / 回合整体力量
#   高端 -> 主动得分（绝杀）    低端 -> 自己失误（下网/出界/吃转）
# 用比值而非绝对强度，是为了抵消球员离麦克风远近、胶皮软硬的差异。
#
# 音频做不到的部分：**归属**。它只知道「这一下很响」，不知道是谁打的。
# 做集锦不需要归属；做训练分析（方案第三阶段）必须有，那绕不开视觉。
RANKERS = {
    "best":    "综合评分（相持长度 + 力量 + 频率 + 运动）",
    "longest": "最长相持 —— 按瞬态数排序",
    "kill":    "强收尾 —— 收尾力量/整体力量 最高（多为主动得分）",
    "weak":    "弱收尾 —— 收尾力量/整体力量 最低（多为自身失误）",
}


def rank(rs: List[Dict], kind: str, cfg: Dict) -> List[Dict]:
    """按不同集锦类型排序（方案模块七的四种类型）。

    弃用的 error 类型：**检不出东西**。判据要求连续 4 次间隔递减且比值一致，
    而检测器准确率仅 0.381，漏掉一次弹跳单调链就断。实测穷尽标注窗内
    收紧时 0 检出，放宽到比值标准差 0.30 时检出 1 个且是假的（准确率 0%%）。
    不要靠放宽阈值让它「有输出」—— 那是在制造结果。
    要修得先把击球检测准确率提上去（见 README 第 9 节的候选重排）。

    关于 error：音频区分不了「失误」和「得分」—— 球下网、出界、对方没接到，
    结局都是球落地弹跳，声学上完全一样。所以这里给的是「回合很快就结束了」，
    它大概率是失误，但也可能是一记好球直接得分。不要当成失误识别。
    """
    if kind == "longest":
        return sorted(rs, key=lambda r: (-r["hits"], -(r["end"] - r["start"])))
    if kind in ("kill", "weak"):
        def ratio(r):
            return r.get("tail_power", 0.0) / max(r.get("power", 0.0), 1e-6)
        # 太短的回合（一两拍）比值噪声大，排除
        cand = [r for r in rs if r["hits"] >= cfg.get("end_min_hits", 4)]
        return sorted(cand, key=ratio, reverse=(kind == "kill"))
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
            m["power"] = max(m["power"], r.get("power", 0.0))
            m["tail_power"] = max(m["tail_power"], r.get("tail_power", 0.0))
        else:
            merged.append({"_a": a, "_b": b, "hits": r["hits"], "score": r["score"],
                           "power": r.get("power", 0.0),
                           "tail_power": r.get("tail_power", 0.0)})

    out = []
    for i, m in enumerate(merged, 1):
        dur = m["_b"] - m["_a"]
        out.append({
            "id": i,
            "start": round(m["_a"], 2),
            "end": round(m["_b"], 2),
            "duration": round(dur, 2),
            "hit_count": m["hits"],
            "power": round(m.get("power", 0.0), 1),
            "tail_power": round(m.get("tail_power", 0.0), 1),
            "hit_rate": round(m["hits"] / max(dur, 1e-6), 2),
            "confidence": round(m["score"], 4),
        })
    return out

"""集锦与剪辑（方案模块六「精彩评分」+ 模块七「自动剪辑」）。

两种相反的取舍，都是有效产品
----------------------------
  * 集锦（best/longest/kill/...）押**准确**：挑出来的都好看，
    但只覆盖一小部分。实测 top10 准确 0.921 / 召回 0.135。
  * 完整版（trim）押**召回**：一个球都不漏，代价是多留些等待。
    实测 pad=0.5 时准确 0.574 / 召回 0.927，12 分钟压到 4:12。

早期我用 F1=0.618 判定「剪空闲做不好」，这个结论下早了 ——
F1 是对称指标，而这个需求是非对称的（漏球比多留难受得多）。
对照商业产品 BetterPlay 剪同一段：保留 64%、准确 0.326、召回 0.962，
它选的正是高召回工作点，而用户接受。
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
            "peak_power": float(av.max()) if len(av) else 0.0,
            # 绝杀用：最后两拍的力量。回合以一记重杀结束才算绝杀
            "tail_power": float(av[-2:].max()) if len(av) else 0.0,
            # 最后一拍的时刻 —— 慢动作要精确对准它，不能靠猜片段中点
            "last_hit": float(ts[-1]),
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
def trim_idle(hits: np.ndarray, duration: float, cfg: Dict) -> List[Dict]:
    """完整版：只剪掉等待/捡球，保留所有打球内容。

    和集锦是**相反的取舍**：集锦押准确（挑出来的都好看，但只覆盖一小部分），
    这里押召回（一个球都不漏，代价是多留一些等待）。

    做法极简：每个候选前后各留 pad 秒，合并重叠。对着真值实测
    （06a6c98c 的 120-240s 窗口，17 个回合）：

    | pad | 保留 | 准确 | 召回 | F1 |
    |---|---|---|---|---|
    | 0.5 | 35% | 0.574 | 0.927 | **0.709** |
    | 1.0 | 52% | 0.417 | 1.000 | 0.589 |
    | 1.5 | 64% | 0.337 | 1.000 | 0.504 |

    对照 BetterPlay 的同一段：保留 64%、准确 0.326、召回 0.962、F1 0.487。
    pad=0.5 时我们用一半的时长覆盖差不多的球。
    """
    pad = float(cfg.get("trim_pad_s", 0.5))
    if len(hits) == 0:
        return []
    merged: List[List[float]] = []
    for t in hits:
        a, b = max(0.0, t - pad), min(duration, t + pad)
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for i, (a, b) in enumerate(merged, 1):
        if b - a < cfg.get("trim_min_s", 0.6):
            continue
        out.append({"id": len(out) + 1, "start": round(a, 2), "end": round(b, 2),
                    "duration": round(b - a, 2), "hit_count": 0,
                    "power": 0.0, "tail_power": 0.0, "last_hit": 0.0,
                    "hit_rate": 0.0, "confidence": 1.0})
    return out


def video_kind(rs: List[Dict], duration: float, cfg: Dict) -> Dict:
    """判断这段录像的结构类型，决定该用什么主题。

    实测三类素材差异极大，同一套主题不可能都合适：
      06a6c98c  空档中位 6.1s  忙碌 19%   两人对练，一半以上时间在捡球
      352e7b27  空档中位 1.2s  忙碌 68%   连续对拉
      dc393956  空档中位 1.5s  忙碌 60%   多球训练
    稀疏型的剪辑价值在「删」（12 分钟压成 1 分钟本身就是产品）；
    密集型没什么可删（压缩比才 2:1），价值在「选」—— 从一堆差不多的球里挑突出的。

    只做二分。再细分（对练 vs 多球）需要知道场上有几个人，那是视觉问题。
    """
    if not rs:
        return {"kind": "unknown", "busy": 0.0, "median_gap": 0.0}
    starts = np.array([r["start"] for r in rs])
    ends = np.array([r["end"] for r in rs])
    gaps = starts[1:] - ends[:-1] if len(rs) > 1 else np.array([0.0])
    busy = float(sum(ends - starts) / max(duration, 1e-6))
    mg = float(np.median(gaps))
    kind = "sparse" if mg >= cfg.get("sparse_gap_s", 4.0) else "dense"
    return {"kind": kind, "busy": round(busy, 3), "median_gap": round(mg, 2)}


# 「自动」曾经按素材结构挑 2-3 个主题集锦（sparse -> best/longest/kill，
# dense -> power/longest/kill）。改成直接等于 trim：用户要的默认结果是
# 「把捡球和等待去掉、一个球都不漏」，而不是替他挑主题。
# 集锦仍然可以单独选，只是不再是默认。

# 给前端用的展示信息：名称、一句话说明、以及适合哪种素材。
# applicable 为空表示通用；否则前端应在结构不匹配时给出提示而不是直接隐藏 ——
# 用户在多球训练视频上选「训练集锦」不会报错，只是压缩比低（约 2:1），
# 应该告诉他「这段素材本来就几乎全在打球，删不掉多少」。
THEMES = [
    {"id": "best",    "name": "训练集锦", "desc": "综合评分最高的若干回合，成片较长",
     "name_en": "Training Reel", "desc_en": "The highest-scoring rallies; longer cut",
     "applicable": ["sparse"]},
    {"id": "longest", "name": "最长对拉", "desc": "来回拍数最多的相持",
     "name_en": "Longest Rally", "desc_en": "The rally with the most exchanges",
     "applicable": []},
    {"id": "kill",    "name": "最帅击球", "desc": "一板打死对手的终结球（收尾力量/回合整体力量 最高）",
     "name_en": "Best Shot", "desc_en": "The winner that ends the rally (tail power / rally power)",
     "applicable": []},
    {"id": "power",   "name": "最重扣杀", "desc": "单拍绝对力量最大的球",
     "name_en": "Hardest Hit", "desc_en": "The single most powerful stroke",
     "applicable": ["dense"]},
    # 命名是产品决定。技术上它只能说明「这一板收尾比该回合平均软」，
    # 推不出是失误还是轻挡得分 —— 所以 desc 保持如实描述，不跟着名字一起夸大。
    {"id": "weak",    "name": "失误合集", "desc": "收尾力量明显低于回合平均的球，多为自身失误",
     "name_en": "Misses", "desc_en": "Rallies ending well below their own average power",
     "applicable": []},
    # 和「训练集锦」的区别要写清楚，否则两个名字听起来都像「最好的部分」，
    # 用户不知道该选哪个：精彩瞬间是三项纪录各一段（很短），训练集锦是综合排名（较长）
    {"id": "trim",    "name": "完整版",
     "name_en": "Full Cut",
     "desc": "只剪掉等待和捡球，一个球都不漏（约压到三分之一）",
     "desc_en": "Removes only waiting and ball-fetching; keeps every rally (~1/3 the length)",
     "applicable": ["sparse"]},
    {"id": "records", "name": "精彩瞬间",
     "name_en": "Top Moments",
     "desc": "全场三项纪录各一段：最长相持、最强击球、最帅收尾（通常 10-30 秒）",
     "desc_en": "Three records, one clip each: longest rally, hardest hit, best finish (10-30s)",
     "applicable": []},
]

RANKERS = {
    "best":    "综合评分（相持长度 + 力量 + 频率 + 运动）",
    "longest": "最长相持 —— 按瞬态数排序",
    "kill":    "最帅击球 —— 收尾力量/整体力量 最高，一板打死对手",
    "weak":    "失误合集 —— 收尾力量/整体力量 最低",
    "power":   "最重扣杀 —— 按回合内单拍绝对力量排序",
    "trim":    "完整版 —— 只剪掉等待，押召回不押准确",
    "records": "精彩瞬间 —— 全场三项纪录各一段",
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
    # 「失误球 = 只有一次击球、随后没有回击」—— 这个规则本身是对的，
    # 真值里 39 个回合有 5 个是单拍（13%），确实是真实存在的类别。
    # 但检测器实现不了：实测检出 15 个单拍回合只有 2 个是真的（准确率 0.13），
    # 真值 12 个只找到 2 个（召回 0.17）。双向都坏 ——
    # 误报候选凭空造出假单拍，真单拍旁边混进误报又变成多拍。
    # 根因和落地弹跳判据一样：准确率 0.6 的候选流里，「孤立事件」对噪声最敏感。
    # 要做这个类别，得先把击球准确率提到 0.9 以上。
    if kind == "records":
        # 每项纪录各取一段。可能撞车（同一回合既最长又最强），去重后按时间排
        if not rs:
            return []
        peak = [r.get("peak_power", 0.0) for r in rs]
        tailr = [r.get("tail_power", 0.0) / max(r.get("power", 1.0), 1e-6) for r in rs]
        idx = {int(np.argmax([r["hits"] for r in rs])),
               int(np.argmax(peak)), int(np.argmax(tailr))}
        return [rs[i] for i in sorted(idx)]
    if kind == "power":
        # 密集素材里回合长度都差不多，能拉开差距的是单拍的绝对力量
        return sorted(rs, key=lambda r: -r.get("peak_power", r.get("power", 0.0)))
    if kind in ("kill", "weak"):
        def ratio(r):
            return r.get("tail_power", 0.0) / max(r.get("power", 0.0), 1e-6)
        # 太短的回合（一两拍）比值噪声大，排除
        cand = [r for r in rs if r["hits"] >= cfg.get("end_min_hits", 4)]
        return sorted(cand, key=ratio, reverse=(kind == "kill"))
    return sorted(rs, key=lambda r: -r["score"])


def dedupe(picked: List[Dict], used: List[Dict], cfg: Dict) -> List[Dict]:
    """去掉与已出片主题重叠的片段。

    一次生成多个主题时，同一段可能在两类里都排前面 —— 用户看到重复内容
    会觉得分类没意义。实测「最帅击球」和「最长对拉」前 10 名重叠 3-6/10
    （力量与回合长度天然相关：回合越长，出现重击的机会越多），
    所以去重是必要的，光换指标解决不了。
    """
    if not used:
        return picked
    out = []
    for p in picked:
        # 多素材时必须先比来源：不同视频的时间轴各自从 0 开始，
        # 只比时间会把两段毫无关系的内容判成重复。
        if not any(p.get("src") == u.get("src")
                   and min(p["end"], u["end"]) - max(p["start"], u["start"])
                   > cfg.get("dedupe_overlap_s", 0.5) for u in used):
            out.append(p)
    for i, p in enumerate(out, 1):
        p["id"] = i
    return out


def select(rs: List[Dict], cfg: Dict, total_s: Optional[float] = None) -> List[Dict]:
    """取前 N 个（或凑满目标时长），再按时间顺序排列 —— 集锦按时间线看更自然。

    这里的 clip_min_s 和 rallies() 的 min_duration_s 是两回事：
    后者是「算不算一个回合」（对着真值调，越准越好，0.3 秒的短回合也算），
    前者是「值不值得剪成一段」（0.3 秒的片段没法看）。
    统计要准，出片要能看，两个目标不同，阈值不能共用。
    """
    pad_a, pad_b = cfg["pad_start_s"], cfg["pad_end_s"]
    clip_min = float(cfg.get("clip_min_s", 0.0))
    if clip_min > 0:
        rs = [r for r in rs if r["end"] - r["start"] >= clip_min]
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
            m["peak_power"] = max(m["peak_power"], r.get("peak_power", 0.0))
            m["last_hit"] = max(m["last_hit"], r.get("last_hit", 0.0))
        else:
            merged.append({"_a": a, "_b": b, "hits": r["hits"], "score": r["score"],
                           "power": r.get("power", 0.0),
                           "tail_power": r.get("tail_power", 0.0),
                           "peak_power": r.get("peak_power", 0.0),
                           "last_hit": r.get("last_hit", 0.0)})

    # 成片最后一段多留一点：整片在最后一拍后立刻黑屏很仓促。
    # 只动最后一段而不是全局加长 —— 全局 +1 秒会让真打球占比从 74% 掉到 60%。
    # 可以为负 —— 用来把过长的结尾收回去。但不能收到比核心还短，
    # 否则最后一拍本身会被切掉。
    extra = float(cfg.get("final_pad_end_s", 0.0))
    if extra and merged:
        last = max(merged, key=lambda m: m["_b"])
        last["_b"] = max(last["_a"] + cfg.get("clip_min_s", 1.0), last["_b"] + extra)

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
            "last_hit": round(m.get("last_hit", 0.0), 3),
            "hit_rate": round(m["hits"] / max(dur, 1e-6), 2),
            "confidence": round(m["score"], 4),
        })
    return out

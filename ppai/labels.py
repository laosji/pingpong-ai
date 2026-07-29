"""标注格式与读写。

设计取舍：主标注是**区间级**（打球/没打球），不是逐拍级。
理由有三：
  1. 区间级就是模块四要的产品指标，也是方案「第一阶段训练目标」的形式；
  2. 一个视频几分钟标完，逐拍标要点几千次；
  3. 区间级足够训练分类器 —— 把区间切成 1 秒窗即为样本。
逐拍标注（hits）是可选的，留给模块五算回合拍数时再补。

阴性素材（没有乒乓球的视频）是免费真值：整段 not_playing，零人工。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

SCHEMA = 1


def empty(video: str, duration: float, complete: bool = False) -> Dict:
    return {
        "schema": SCHEMA,
        "video": os.path.abspath(video),
        "duration": round(duration, 3),
        # complete=True 表示整段都看过了。整片穷尽标注很费时，实际很少这么做。
        "complete": complete,
        # 更实用的形式：只声明**某几段**是穷尽标注的。
        # 准确率只能在这些区间内计算 —— 区间外没有标记不代表那里没发生，
        # 把它当负样本会把漏标算成误报，得出的准确率毫无意义。
        # 召回则不受影响，全部标记都可用。
        "complete_ranges": [],
        # 用户在产品里主动否定的区间（「这段不对」）。与 complete_ranges 分开：
        # 后者零击球会被判为标注遗漏并跳过，而这里的零击球是**用户明确声称**的，
        # 是可信的纯负例。混在一起会让保护机制把真实反馈也挡掉。
        "negative_ranges": [],
        "playing": [],      # [[start, end], ...] 击球/有效比赛
        # 捡球区间：球落地后到重新开始之间。和 playing 分开存 ——
        # 这是**负例**，混进 playing 会让真值直接失效。
        "pickup": [],
        "hits": [],         # 可选：击球瞬态时间点
        # 发球时间点。回合边界现在靠「静音间隔 > gap_s」猜，实测很脆：
        # 34 个长回合里 27 个的内部最大间隔都在阈值 70% 以上，
        # 阈值从 0.65 降到 0.60，最长回合就从 39 拍裂成 29 拍。
        # 发球是回合起点的真值，标它比穷尽标击球便宜约 10 倍
        # （11 次/分钟 vs 100 次/分钟）。
        "serves": [],
        "notes": "",
    }


def negative(video: str, duration: float, note: str = "阴性对照：无乒乓球") -> Dict:
    """整段判定为没打球。用于有声书、影视剧等对照素材。"""
    lab = empty(video, duration, complete=True)
    lab["notes"] = note
    return lab


def path_for(video: str, label_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(label_dir, stem + ".json")


def load(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        lab = json.load(f)
    if lab.get("schema") != SCHEMA:
        raise ValueError("标注格式版本不符: %s" % path)
    lab["playing"] = _normalize(lab.get("playing", []), lab["duration"])
    lab["pickup"] = _normalize(lab.get("pickup", []), lab["duration"])
    lab["negative_ranges"] = _normalize(lab.get("negative_ranges", []), lab["duration"])
    lab["complete_ranges"] = _normalize(lab.get("complete_ranges", []), lab["duration"])
    if lab.get("complete") and not lab["complete_ranges"]:
        lab["complete_ranges"] = [[0.0, lab["duration"]]]
    return lab


def scored_ranges(lab: Dict) -> List[List[float]]:
    """可用于算准确率的时段。空列表表示这份标注只能算召回。"""
    return lab.get("complete_ranges") or []


def save(lab: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lab["playing"] = _normalize(lab.get("playing", []), lab["duration"])
    lab["pickup"] = _normalize(lab.get("pickup", []), lab["duration"])
    lab["negative_ranges"] = _normalize(lab.get("negative_ranges", []), lab["duration"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lab, f, ensure_ascii=False, indent=2)
    return path


def _normalize(intervals: List, duration: float) -> List[List[float]]:
    """排序、裁剪到时长内、合并重叠。手工拖出来的区间常常有重叠。"""
    out: List[List[float]] = []
    for iv in sorted([[max(0.0, float(a)), min(duration, float(b))] for a, b in intervals]):
        if iv[1] - iv[0] <= 1e-6:
            continue
        if out and iv[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], iv[1])
        else:
            out.append(iv)
    return [[round(a, 3), round(b, 3)] for a, b in out]


def to_mask(intervals: List, duration: float, dt: float) -> np.ndarray:
    """区间 -> 时间网格上的布尔掩码，评测用。"""
    n = max(1, int(np.ceil(duration / dt)))
    mask = np.zeros(n, dtype=bool)
    for a, b in intervals:
        mask[int(a / dt):int(np.ceil(b / dt))] = True
    return mask


def find(video: str, label_dir: str) -> Optional[Dict]:
    p = path_for(video, label_dir)
    return load(p) if os.path.exists(p) else None

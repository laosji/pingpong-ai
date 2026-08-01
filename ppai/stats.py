"""训练数据战报（方案模块五的统计部分 + 商业化里的「训练记录」）。

所有数字都基于击球检测，而它当前准确率约 0.38、召回 0.85 —— 所以
「总击球数」这类绝对量会**系统性偏高**（误报计入）。报告里如实标注，
不要让用户以为这是精确计数。相对量（最长相持排第几、力量对比）
受影响小得多，因为误报是弥散的、对各回合大致同等影响。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def playing_time(hits: np.ndarray, gap: float) -> float:
    """有效打球时长。**用比回合分组更小的 gap** —— 两个目标互相冲突：

    真实回合只有 0.5 秒左右，只要有一个误报落在最后一拍后 gap 秒内，
    回合长度就翻倍。实测（对着 5 个穷尽标注窗口）：
        gap 0.3 -> 忙碌占比误差 5.1 个百分点，回合数误差 7.6 个
        gap 0.5 -> 13.1 个百分点，2.4 个
        gap 0.7 -> 15.4 个百分点，2.4 个
    没有一个值能同时做准 —— 小 gap 时长准但回合被切碎，大 gap 反之。
    所以时长用 0.3、回合分组用 0.7，各取所长。
    """
    if len(hits) < 2:
        return 0.0
    total, start, prev = 0.0, hits[0], hits[0]
    for t in hits[1:]:
        if t - prev > gap:
            total += prev - start
            start = t
        prev = t
    return float(total + prev - start)


def summarize(rs: List[Dict], hits: np.ndarray, amps: np.ndarray,
              duration: float, kind: Dict, busy_gap: float = 0.3) -> Dict:
    if not rs:
        return {"duration": duration, "rally_count": 0}

    n = np.array([r["hits"] for r in rs])
    dur = np.array([r["end"] - r["start"] for r in rs])
    peak = np.array([r.get("peak_power", 0.0) for r in rs])
    play_s = playing_time(np.asarray(hits), busy_gap)

    i_long = int(np.argmax(n))
    i_pow = int(np.argmax(peak))

    return {
        "duration": round(duration, 1),
        "structure": kind["kind"],
        "busy_ratio": kind["busy"],
        "rally_count": len(rs),
        "transient_count": int(len(hits)),
        "playing_seconds": round(play_s, 1),
        "playing_ratio": round(play_s / max(duration, 1e-6), 3),
        "rally_len_median": int(np.median(n)),
        "rally_len_mean": round(float(n.mean()), 1),
        "rally_dur_median": round(float(np.median(dur)), 1),
        "records": {
            "longest_rally": {"index": i_long, "hits": int(n[i_long]),
                              "start": rs[i_long]["start"], "end": rs[i_long]["end"],
                              "duration": round(float(dur[i_long]), 1)},
            "strongest_hit": {"index": i_pow, "power": round(float(peak[i_pow]), 1),
                              "start": rs[i_pow]["start"], "end": rs[i_pow]["end"]},
        },
    }


def _mmss(x: float) -> str:
    return "%d:%02d" % (int(x) // 60, int(x) % 60)


def report(s: Dict) -> None:
    if not s.get("rally_count"):
        print("  没有检出回合")
        return
    print("  时长 %s | 结构 %s | 有效打球 %s（%.0f%%）"
          % (_mmss(s["duration"]),
             "稀疏（大量捡球）" if s["structure"] == "sparse" else "密集（连续击球）",
             _mmss(s["playing_seconds"]), s["playing_ratio"] * 100))
    print("  回合 %d 个 | 每回合中位 %d 个瞬态 / %.1f 秒"
          % (s["rally_count"], s["rally_len_median"], s["rally_dur_median"]))
    r = s["records"]
    print("\n  全场之最")
    print("    最长相持   %s-%s  %d 个瞬态，%.1f 秒"
          % (_mmss(r["longest_rally"]["start"]), _mmss(r["longest_rally"]["end"]),
             r["longest_rally"]["hits"], r["longest_rally"]["duration"]))
    print("    最强击球   %s-%s  力量 %.0f"
          % (_mmss(r["strongest_hit"]["start"]), _mmss(r["strongest_hit"]["end"]),
             r["strongest_hit"]["power"]))
    print("\n  注：瞬态数不等于挥拍数 —— 检测器准确率约 0.38，"
          "「%d 个瞬态」这类绝对量偏高。" % s["transient_count"])
    print("      排名和倍数关系受影响小得多（误报是弥散的，对各回合大致同等影响）。")

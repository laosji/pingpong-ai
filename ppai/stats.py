"""训练数据战报（方案模块五的统计部分 + 商业化里的「训练记录」）。

所有数字都基于击球检测，而它当前准确率约 0.38、召回 0.85 —— 所以
「总击球数」这类绝对量会**系统性偏高**（误报计入）。报告里如实标注，
不要让用户以为这是精确计数。相对量（最长相持排第几、力量对比）
受影响小得多，因为误报是弥散的、对各回合大致同等影响。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def summarize(rs: List[Dict], hits: np.ndarray, amps: np.ndarray,
              duration: float, kind: Dict) -> Dict:
    if not rs:
        return {"duration": duration, "rally_count": 0}

    n = np.array([r["hits"] for r in rs])
    dur = np.array([r["end"] - r["start"] for r in rs])
    peak = np.array([r.get("peak_power", 0.0) for r in rs])
    tail = np.array([r.get("tail_power", 0.0) for r in rs])
    play_s = float(dur.sum())

    i_long = int(np.argmax(n))
    i_pow = int(np.argmax(peak))
    i_tail = int(np.argmax(tail / np.maximum(
        np.array([r.get("power", 1.0) for r in rs]), 1e-6)))

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
            "best_finish": {"index": i_tail,
                            "tail_power": round(float(tail[i_tail]), 1),
                            "ratio": round(float(tail[i_tail] /
                                                 max(rs[i_tail].get("power", 1.0), 1e-6)), 2),
                            "start": rs[i_tail]["start"], "end": rs[i_tail]["end"]},
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
    print("\n  单项之最")
    print("    最长相持   %s-%s  %d 个瞬态，%.1f 秒"
          % (_mmss(r["longest_rally"]["start"]), _mmss(r["longest_rally"]["end"]),
             r["longest_rally"]["hits"], r["longest_rally"]["duration"]))
    print("    最强击球   %s-%s  力量 %.0f"
          % (_mmss(r["strongest_hit"]["start"]), _mmss(r["strongest_hit"]["end"]),
             r["strongest_hit"]["power"]))
    print("    最强收尾   %s-%s  收尾/整体 %.2f 倍"
          % (_mmss(r["best_finish"]["start"]), _mmss(r["best_finish"]["end"]),
             r["best_finish"]["ratio"]))
    print("\n  注：瞬态数不等于挥拍数 —— 检测器准确率约 0.38，"
          "「%d 个瞬态」这类绝对量偏高。" % s["transient_count"])
    print("      排名和倍数关系受影响小得多（误报是弥散的，对各回合大致同等影响）。")

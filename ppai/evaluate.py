"""对着人工标注算准确率。把「猜阈值」变成「测准确率」。

指标选择：帧级（0.1 秒网格）的 playing 类 precision / recall / F1。
不用片段级 IoU 作为主指标 —— 一个回合被切成两段这种错误，
帧级几乎无损失，但用户体验上无所谓；而把休息时间判成打球，帧级会如实惩罚。
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List

import numpy as np

from . import labels as L


def _prf(pred: np.ndarray, truth: np.ndarray) -> Dict:
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def evaluate(videos: List[str], label_dir: str, predict: Callable[[str], List[Dict]],
             dt: float = 0.1) -> Dict:
    """predict(video) -> segments。返回逐文件与汇总指标。"""
    rows, agg = [], {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    skipped = []

    for v in videos:
        lab = L.find(v, label_dir)
        if lab is None:
            skipped.append((os.path.basename(v), "无标注"))
            continue
        scored = L.scored_ranges(lab)
        if not scored:
            skipped.append((os.path.basename(v),
                            "没有 complete_ranges —— 抽样标注只能算召回，算准确率会把漏标当误报"))
            continue

        dur = lab["duration"]
        truth = L.to_mask(lab["playing"], dur, dt)
        pred = L.to_mask([[s["start"], s["end"]] for s in predict(v)], dur, dt)
        n = min(len(pred), len(truth))
        # 只在声明为穷尽标注的时段内计分
        keep = L.to_mask(scored, dur, dt)[:n]
        m = _prf(pred[:n][keep], truth[:n][keep])
        m["name"] = os.path.basename(v)[:34]
        m["duration"] = dur
        m["truth_ratio"] = float(truth[:n][keep].mean())
        m["pred_ratio"] = float(pred[:n][keep].mean())
        rows.append(m)
        for k in agg:
            agg[k] += m[k]

    total = sum(agg.values())
    overall = _prf(np.array([True] * agg["tp"] + [True] * agg["fp"] +
                            [False] * agg["fn"] + [False] * agg["tn"]),
                  np.array([True] * agg["tp"] + [False] * agg["fp"] +
                           [True] * agg["fn"] + [False] * agg["tn"])) if total else {}
    return {"rows": rows, "overall": overall, "skipped": skipped}


def report(res: Dict) -> None:
    if res["skipped"]:
        print("跳过：")
        for name, why in res["skipped"]:
            print("  %-36s %s" % (name, why))
        print()

    if not res["rows"]:
        print("没有可评测的样本。先用 `annotate` 生成标注工具，或用 `label-negative` 标阴性素材。")
        return

    print("%-36s %7s %7s %8s %8s %6s" % ("文件", "真实占比", "预测占比", "准确率", "召回率", "F1"))
    for r in res["rows"]:
        print("%-36s %6.0f%% %6.0f%% %7.2f %7.2f %6.2f"
              % (r["name"], r["truth_ratio"] * 100, r["pred_ratio"] * 100,
                 r["precision"], r["recall"], r["f1"]))

    o = res["overall"]
    print("\n汇总  准确率 %.3f  召回率 %.3f  F1 %.3f" % (o["precision"], o["recall"], o["f1"]))
    print("      误报 %d 帧 / 漏报 %d 帧（0.1 秒一帧）" % (o["fp"], o["fn"]))
    # MVP 成功标准：有效片段识别 80%
    if o["f1"] >= 0.80:
        print("      达到方案 MVP 指标（有效片段识别 80%）")
    else:
        print("      未达方案 MVP 指标 0.80")


def hit_level(det: "np.ndarray", truth: "np.ndarray", ranges, tol: float = 0.35) -> Dict:
    """击球级评测，只在 ranges（穷尽标注时段）内计分。

    匹配用**容差判定**而非一一配对：
      * 召回 —— 每个人工标记 ±tol 内有检出即算命中
      * 准确 —— 每个检出 ±tol 内有任一人工标记即算正确
    这样人只需标「球拍击中球」，紧随其后的台面弹跳落在同一容差窗内，
    不会被当成误报。一一配对会把弹跳算错，从而低估准确率。
    """
    import numpy as np

    def inside(ts):
        m = np.zeros(len(ts), bool)
        for a, b in ranges:
            m |= (ts >= a) & (ts <= b)
        return m

    det = np.asarray(det)[inside(np.asarray(det))] if len(det) else np.zeros(0)
    truth = np.asarray(truth)[inside(np.asarray(truth))] if len(truth) else np.zeros(0)
    if len(truth) == 0:
        return {"n_truth": 0, "n_det": len(det), "recall": float("nan"),
                "precision": float("nan"), "f1": float("nan")}

    hit_t = np.array([np.any(np.abs(det - t) <= tol) for t in truth]) if len(det) \
        else np.zeros(len(truth), bool)
    ok_d = np.array([np.any(np.abs(truth - x) <= tol) for x in det]) if len(det) \
        else np.zeros(0, bool)

    rec = float(hit_t.mean())
    prec = float(ok_d.mean()) if len(det) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else float("nan")
    return {"n_truth": len(truth), "n_det": len(det),
            "recall": rec, "precision": prec, "f1": f1}

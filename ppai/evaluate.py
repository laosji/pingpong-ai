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
        if not lab.get("complete"):
            skipped.append((os.path.basename(v), "标注未完成，算召回会失真"))
            continue

        dur = lab["duration"]
        truth = L.to_mask(lab["playing"], dur, dt)
        pred = L.to_mask([[s["start"], s["end"]] for s in predict(v)], dur, dt)
        n = min(len(pred), len(truth))
        m = _prf(pred[:n], truth[:n])
        m["name"] = os.path.basename(v)[:34]
        m["duration"] = dur
        m["truth_ratio"] = float(truth.mean())
        m["pred_ratio"] = float(pred.mean())
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

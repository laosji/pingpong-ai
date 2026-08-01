"""端到端对照：换掉重排特征之后，**回合排序**变好还是变坏。

为什么不能只看候选级 AUC
------------------------
这个项目栽过一次：把 k_mad 调高让击球级 F1 涨了，却把回合排序的 ρ
从 0.79 打到 0.64。候选准确率和成片质量之间隔着「分组」这一层，
分组对候选的**分布**敏感，不只对数量敏感。

所以这里量的是产品真正依赖的那个量：**检测出来的回合，按长度排序，
和人工真值的回合长度排序有多一致（Spearman ρ）**。
README 里现役管线的这个数是 0.844 -> 0.906（启用发球门禁后）。

配对方式：真值回合和检测回合按时间重叠配对，重叠 < 0.2 秒不算 ——
这条是有来历的，早先用「重叠 > 0」得出过「top-3 横跨两个回合」的假结论。
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppai import audio, config, highlight, labels as L, rerank  # noqa: E402
import hires_onset as H  # noqa: E402

MIN_OVERLAP = 0.2


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """自己算，避免为一个相关系数引 scipy（app 里已经把 scipy 去掉了）。"""
    if len(a) < 3:
        return float("nan")
    def rank(x):
        o = np.argsort(x)
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x), dtype=float)
        # 并列取平均秩，否则相同长度的回合会被人为分出先后
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        if cnt.max() > 1:
            for v in np.flatnonzero(cnt > 1):
                m = inv == v
                r[m] = r[m].mean()
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def rank_rho(truth_hits: np.ndarray, det_hits: np.ndarray, dur: float, hcfg: Dict) -> Dict:
    """真值回合 vs 检测回合的长度排序一致性。"""
    tr = highlight.rallies_gated(np.asarray(truth_hits, float), dur, hcfg)
    de = highlight.rallies_gated(np.asarray(det_hits, float), dur, hcfg)
    if len(tr) < 3 or len(de) < 3:
        return {"rho": float("nan"), "n": min(len(tr), len(de)),
                "n_true": len(tr), "n_det": len(de)}
    a, b = [], []
    used = set()
    for t in tr:
        best, bo = None, MIN_OVERLAP
        for j, d in enumerate(de):
            if j in used:
                continue
            ov = min(t["end"], d["end"]) - max(t["start"], d["start"])
            if ov > bo:
                best, bo = j, ov
        if best is not None:
            used.add(best)
            a.append(t["hits"]); b.append(de[best]["hits"])
    return {"rho": spearman(a, b), "n": len(a), "n_true": len(tr), "n_det": len(de)}


def main():
    cfg = config.load()
    acfg, hcfg = cfg["audio"], cfg["highlight"]
    keep_ratio = cfg.get("rerank", {}).get("keep_ratio", 0.6)
    half = int(os.environ.get("PIPO_HALF", "100"))
    H.PATCH_HALF = half

    # 收集所有标注视频的候选、log-mel、PANNs 特征
    items = []
    for path in sorted(glob.glob("labels/*.json")):
        lab = L.load(path)
        rng = L.scored_ranges(lab)
        if not rng or not lab.get("playing"):
            continue
        v = lab["video"]
        if not os.path.exists(v):
            continue
        pcm = audio.extract_pcm(v, acfg["sr"])
        det, env, _, fr = audio.detect_hits(pcm, acfg)
        a0, b0 = rng[0][0], rng[-1][1]
        m = (det >= a0) & (det <= b0)
        cand = det[m]
        truth = np.array(sorted((a + b) / 2.0 for a, b in lab["playing"]))
        truth = truth[(truth >= a0) & (truth <= b0)]
        yy = np.array([1 if np.any(np.abs(truth - c) <= rerank.TOL) else 0 for c in cand])
        lm = H.logmel(pcm, acfg)
        fi = np.clip((cand * fr).round().astype(int), 0, max(lm.shape[1] - 1, 0))
        items.append({
            "name": os.path.basename(path), "cand": cand, "y": yy, "truth": truth,
            "dur": b0 - a0, "off": a0,
            "mel": H.patches(lm, fi).reshape(len(cand), -1),
            "pan": rerank._features(v, cand, "cache"),
        })
        print("  %-38s %4d 候选 / %3d 真" % (os.path.basename(path)[:38], len(cand), yy.sum()))

    names = [it["name"] for it in items]
    print("\n  留一视频交叉验证，每次用另外几个视频训练重排器\n")
    print("  %-30s %8s %8s %8s" % ("留出", "不重排", "PANNs", "log-mel"))
    rows = []
    for i, it in enumerate(items):
        tr = [j for j in range(len(items)) if j != i]
        ytr = np.concatenate([items[j]["y"] for j in tr])
        gtr = np.concatenate([np.full(len(items[j]["y"]), names[j]) for j in tr])
        if len(np.unique(it["y"])) < 2:
            continue

        base = rank_rho(it["truth"], it["cand"], it["dur"], hcfg)

        out = {}
        for key in ("pan", "mel"):
            from sklearn.linear_model import LogisticRegression
            X = np.vstack([items[j][key] for j in tr])
            c = LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced")\
                .fit(X, ytr, sample_weight=H._balance(gtr))
            p = c.predict_proba(it[key])[:, 1]
            k = max(1, int(round(len(p) * keep_ratio)))
            kept = np.sort(it["cand"][np.argsort(-p)[:k]])
            out[key] = rank_rho(it["truth"], kept, it["dur"], hcfg)

        rows.append((base, out["pan"], out["mel"]))
        print("  %-30s %8.3f %8.3f %8.3f" %
              (it["name"][:30], base["rho"], out["pan"]["rho"], out["mel"]["rho"]))

    if rows:
        m = np.array([[r[0]["rho"], r[1]["rho"], r[2]["rho"]] for r in rows], float)
        avg = np.nanmean(m, axis=0)
        print("\n  平均 ρ      不重排 %.3f | PANNs %.3f | log-mel±%dms %.3f"
              % (avg[0], avg[1], half * 10, avg[2]))
        d = avg[2] - avg[1]
        print("\n  → %s" % (
            "log-mel 端到端也更好（+%.3f），可以换" % d if d > 0.02 else
            ("两者端到端基本持平（差 %+.3f）—— 那就选便宜的那个：log-mel "
             "不需要 323MB 预训练模型" % d) if abs(d) <= 0.02 else
            "**log-mel 端到端更差（%+.3f）**，候选级 AUC 的优势没有传导过来，不能换" % d))


if __name__ == "__main__":
    main()

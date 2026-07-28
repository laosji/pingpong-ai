"""候选重排 —— 把击球检测的准确率从 0.38-0.49 提到 0.58-0.68。

为什么是重排而不是重做检测
--------------------------
谱通量检测器的**召回是 0.850**，真击球 85% 已经在候选里，问题只是
候选里混了一半以上的假货。所以不需要重新做检测，只需要在候选上做二分类。

跨视频有效性（已验证）
----------------------
用 A 训练、在 B 上测（两段素材球馆/灯光/训练形式全不同）：

| 训练→测试 | 保留比例 | 准确 | 召回 | F1 |
|---|---|---|---|---|
| 06a6c98c→dc393956 | 100%（基线） | 0.493 | 1.000 | 0.660 |
|                   | 60%          | 0.678 | 0.824 | 0.744 |
| dc393956→06a6c98c | 100%（基线） | 0.381 | 1.000 | 0.552 |
|                   | 50%          | 0.580 | 0.755 | 0.656 |

**必须按比例保留，不能用固定概率阈值。** 概率标定在不同视频上差异很大：
B→A 用固定阈值 0.5 时召回只剩 0.170，而按比例保留 50% 时召回 0.755。
排序能力跨视频，绝对概率值不跨视频。

另一个反直觉的观察：A→B 的 AUC(0.752) 高于 B 自己内部训练的 AUC(0.717)，
因为 A 的训练数据更多（139 候选 vs 75）。数据量比同源更重要。
"""
from __future__ import annotations

import glob
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

MODEL_PATH = "models/rerank.pkl"
TOL = 0.35        # 候选与人工标记的匹配容差，与 evaluate.hit_level 保持一致


def _features(video: str, cand: np.ndarray, cache_dir: Optional[str]) -> np.ndarray:
    from . import embed
    t, f = embed.embed_video(video, cache_dir=cache_dir)
    if len(t) == 0:
        return np.zeros((len(cand), 0))
    idx = np.array([np.argmin(np.abs(t - c)) for c in cand])
    return f[idx]


def build_dataset(label_dir: str, cfg: Dict, cache_dir: str = "cache") -> Tuple:
    """从所有带 complete_ranges 的标注里构造训练集。"""
    from . import audio, labels as L

    X, y, groups = [], [], []
    for path in sorted(glob.glob(os.path.join(label_dir, "*.json"))):
        lab = L.load(path)
        rng = L.scored_ranges(lab)
        if not rng or not lab["playing"]:
            continue
        video = lab["video"]
        if not os.path.exists(video):
            print("  跳过（视频不存在）: %s" % os.path.basename(video))
            continue
        # 只标了捡球、没标击球的区间，是标注遗漏而不是「这段真的没人打球」——
        # 若当成穷尽标注，该区间所有候选（含真实击球）都会变成负例，
        # 比不用这个区间更糟。阴性对照视频没有 pickup 标注，不会被误伤。
        bad = [r for r in rng
               if not any(r[0] <= (a + b) / 2 <= r[1] for a, b in lab["playing"])
               and any(r[0] <= a <= r[1] for a, b in lab["pickup"])]
        if bad:
            for a, b in bad:
                print("  ⚠️  %s 的 %.0f-%.0fs 标了捡球但零击球，判定为标注遗漏，已跳过"
                      % (os.path.basename(path), a, b))
            rng = [r for r in rng if r not in bad]
            if not rng:
                continue

        pcm = audio.extract_pcm(video, cfg["audio"]["sr"])
        det, _, _, _ = audio.detect_hits(pcm, cfg["audio"])
        keep = np.zeros(len(det), bool)
        for a, b in rng:
            keep |= (det >= a) & (det <= b)
        cand = det[keep]
        if len(cand) == 0:
            continue
        truth = np.array(sorted((a + b) / 2.0 for a, b in lab["playing"]))
        yy = np.array([1 if np.any(np.abs(truth - c) <= TOL) else 0 for c in cand])
        X.append(_features(video, cand, cache_dir))
        y.append(yy)
        groups.append(np.full(len(cand), os.path.basename(path)))
        print("  %-40s %4d 候选 / %3d 真 (准确率 %.3f)"
              % (os.path.basename(video)[:40], len(cand), yy.sum(), yy.mean()))
    if not X:
        raise SystemExit("没有可用标注：需要 complete_ranges 且有 playing 标记")
    return np.vstack(X), np.concatenate(y), np.concatenate(groups)


def _balance(groups: np.ndarray) -> np.ndarray:
    """按视频均衡的样本权重：每个视频的总权重相同。

    实测必要性：某个视频标得多（427 候选 vs 139/150）时会主导训练，
    其它视频反而变差 —— dc393956 的 AUC 从 0.782 掉到 0.712、准确率 0.689->0.589。
    加权后恢复到 0.741 / 0.678。「数据越多越好」只在均衡时成立。
    """
    w = np.ones(len(groups), float)
    srcs = set(groups)
    for s in srcs:
        m = groups == s
        w[m] = len(groups) / (len(srcs) * m.sum())
    return w


def train(label_dir: str, cfg: Dict, out_path: str = MODEL_PATH,
          cache_dir: str = "cache") -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y, g = build_dataset(label_dir, cfg, cache_dir)
    files = sorted(set(g))
    print("\n合计 %d 候选 / %d 真，来自 %d 份标注" % (len(y), y.sum(), len(files)))

    # 留一视频交叉验证 —— 必须按文件留出，否则同一视频的相邻候选会泄漏
    if len(files) > 1:
        print("\n留一视频交叉验证：")
        for f in files:
            tr, te = g != f, g == f
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            c = LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced")\
                .fit(X[tr], y[tr], sample_weight=_balance(g[tr]))
            p = c.predict_proba(X[te])[:, 1]
            k = int(len(p) * 0.6)
            idx = np.argsort(-p)[:k]
            print("  留出 %-34s AUC %.3f | 保留60%%时 准确 %.3f (基线 %.3f) 召回 %.3f"
                  % (f[:34], roc_auc_score(y[te], p), y[te][idx].mean(),
                     y[te].mean(), y[te][idx].sum() / max(y[te].sum(), 1)))
    else:
        print("\n只有一份标注，无法做跨视频验证 —— 再标一个视频才知道能不能泛化")

    clf = LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced")\
        .fit(X, y, sample_weight=_balance(g))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump({"clf": clf, "n_train": len(y), "files": files}, fh)
    print("\n已保存: %s" % out_path)
    return {"n": len(y), "files": files}


def load(path: str = MODEL_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def apply(video: str, cand: np.ndarray, model, keep_ratio: float = 0.6,
          cache_dir: str = "cache") -> np.ndarray:
    """返回保留下来的候选（按比例，不用固定概率阈值 —— 概率标定不跨视频）。"""
    if model is None or len(cand) == 0:
        return cand
    X = _features(video, cand, cache_dir)
    if X.shape[1] == 0:
        return cand
    p = model["clf"].predict_proba(X)[:, 1]
    k = max(1, int(round(len(cand) * keep_ratio)))
    return np.sort(cand[np.argsort(-p)[:k]])

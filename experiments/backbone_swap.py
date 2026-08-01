"""实验：用 EfficientAT 的 MobileNetV3 换掉 323MB 的 PANNs CNN14。

为什么值得试（调研结论，不是猜测）
----------------------------------
2025 年的设备端评测：CNN14 单次推理 ~3.5 秒、温度冲到 85°C 以上，
作者明确建议生产环境别用；mn05_as / mn10_as 是 ~0.25 秒、55-75°C。

而精度上小模型并不吃亏：

    CNN14      80M 参数   323MB    AudioSet mAP 43.1
    mn04_as   0.98M       ~4MB                 43.2
    mn10_as   4.88M      ~20MB                 47.1

HEAR 基准（19 个下游任务的线性探针评测，和我们「嵌入 + 逻辑回归」
用法一致）上 mn10 也优于 CNN14。

**但这些都不是我们的任务。** 这个项目已经栽过两次「别处的指标传不过来」：
PANNs 跨视频 AUC 0.993 到视频内只剩 0.773；高分辨率 log-mel 候选级赢了
却在端到端等于不重排。所以照旧：同一批候选、同一套留一交叉验证，
再加端到端的回合排序。

两个臂之外还留了 CNN14 当基线，三个数放一起才有意义。
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EFF = os.environ.get(
    "EFFICIENT_AT",
    "/private/tmp/claude-501/-Users-duanchao-Downloads/"
    "f577a54d-6089-4994-be12-45d55cc86df8/scratchpad/EfficientAT")
sys.path.insert(0, EFF)

from ppai import audio, config, labels as L, rerank  # noqa: E402

PANNS_SR = 32000
WIN = 32000        # 1 秒，和 CNN14 的窗一致 —— 不一致就不是同一个对照
HOP = 16000
KEEP = 0.6


def load_mn(name: str):
    """载入 EfficientAT 的 MobileNetV3 + 它自己的 mel 前端。

    要临时切到 EfficientAT 目录：它的 helpers/utils.py 在**导入时**就用
    相对路径读 metadata/class_labels_indices.csv，不切过去直接 FileNotFoundError。
    """
    import torch
    cwd = os.getcwd()
    os.chdir(EFF)
    try:
        from models.mn.model import get_model
        from models.preprocess import AugmentMelSTFT
    finally:
        os.chdir(cwd)

    m = get_model(width_mult={"mn04_as": 0.4, "mn05_as": 0.5,
                              "mn10_as": 1.0, "mn20_as": 2.0}[name],
                  pretrained_name=name)
    m.eval()
    # 推理时关掉 SpecAugment 的两个遮挡（freqm/timem）—— 它们只在训练时用，
    # 留着会让同一段音频每次出不同的嵌入。
    mel = AugmentMelSTFT(n_mels=128, sr=PANNS_SR, win_length=800, hopsize=320,
                         n_fft=1024, freqm=0, timem=0)
    mel.eval()
    return m, mel


def embed_mn(pcm: np.ndarray, m, mel, batch: int = 32) -> np.ndarray:
    """滑窗提嵌入，窗和跳与 CNN14 完全相同。返回 [N, D]。"""
    import torch

    x = pcm if len(pcm) >= WIN else np.pad(pcm, (0, WIN - len(pcm)))
    starts = np.arange(0, len(x) - WIN + 1, HOP)
    out = []
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunk = np.stack([x[s:s + WIN] for s in starts[i:i + batch]])
            spec = mel(torch.from_numpy(chunk).float())
            _, feat = m(spec.unsqueeze(1))
            out.append(feat.numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def embed_cnn14(pcm: np.ndarray) -> np.ndarray:
    from ppai import embed
    return embed.embed_windows(pcm)[1]


def times_of(n: int) -> np.ndarray:
    return (np.arange(n) * HOP + WIN / 2.0) / PANNS_SR


def _balance(g: np.ndarray) -> np.ndarray:
    w = np.ones(len(g), float)
    for s in set(g):
        mm = g == s
        w[mm] = len(g) / (len(set(g)) * mm.sum())
    return w


def collect(arms: List[str]) -> Dict:
    """所有臂共用同一批候选和标签 —— 不共用就不是对照实验。"""
    cfg = config.load()
    acfg = cfg["audio"]
    mn = {a: load_mn(a) for a in arms if a != "cnn14"}
    per = []
    for path in sorted(glob.glob("labels/*.json")):
        lab = L.load(path)
        rng = L.scored_ranges(lab)
        neg = lab.get("negative_ranges") or []
        if not rng and not neg:
            continue
        if rng and not lab["playing"]:
            continue
        v = lab["video"]
        if not os.path.exists(v):
            print("  跳过（原片不在）: %s" % os.path.basename(path))
            continue

        pcm16 = audio.extract_pcm(v, acfg["sr"])
        det, _, _, _ = audio.detect_hits(pcm16, acfg)
        keep = np.zeros(len(det), bool)
        for a, b in rng:
            keep |= (det >= a) & (det <= b)
        in_neg = np.zeros(len(det), bool)
        for a, b in neg:
            in_neg |= (det >= a) & (det <= b)
        keep |= in_neg
        cand = det[keep]
        if len(cand) == 0:
            continue
        truth = (np.array(sorted((a + b) / 2.0 for a, b in lab["playing"]))
                 if lab["playing"] else np.zeros(0))
        y = np.array([0 if in_neg[np.searchsorted(det, c)]
                      else (1 if len(truth) and np.any(np.abs(truth - c) <= rerank.TOL) else 0)
                      for c in cand])

        pcm32 = audio.extract_pcm(v, PANNS_SR)
        feats = {}
        for a in arms:
            f = embed_cnn14(pcm32) if a == "cnn14" else embed_mn(pcm32, *mn[a])
            t = times_of(len(f))
            idx = np.array([np.argmin(np.abs(t - c)) for c in cand])
            feats[a] = f[idx]
        a0, b0 = rng[0][0], rng[-1][1]
        tr_hits = truth[(truth >= a0) & (truth <= b0)] if len(truth) else None
        per.append({"name": os.path.basename(path), "y": y, "feats": feats,
                    "cand": cand, "truth": tr_hits, "dur": b0 - a0})
        print("  %-38s %4d 候选 / %3d 真   维度 %s"
              % (os.path.basename(path)[:38], len(cand), y.sum(),
                 {a: feats[a].shape[1] for a in arms}))
    return {"per": per, "arms": arms}


def lovo(data: Dict) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    per, arms = data["per"], data["arms"]
    names = [p["name"] for p in per]
    y = np.concatenate([p["y"] for p in per])
    g = np.concatenate([np.full(len(p["y"]), p["name"]) for p in per])

    print("\n  留一视频交叉验证（候选级）\n")
    print("  %-30s %s" % ("留出", "  ".join("%-14s" % a for a in arms)))
    acc = {a: [] for a in arms}
    for i, p in enumerate(per):
        if len(np.unique(p["y"])) < 2:
            continue
        row = []
        for a in arms:
            X = np.vstack([q["feats"][a] for q in per])
            tr, te = g != p["name"], g == p["name"]
            c = LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced")\
                .fit(X[tr], y[tr], sample_weight=_balance(g[tr]))
            pr = c.predict_proba(X[te])[:, 1]
            k = max(1, int(len(pr) * KEEP))
            auc = roc_auc_score(y[te], pr)
            prec = float(y[te][np.argsort(-pr)[:k]].mean())
            acc[a].append((auc, prec))
            row.append("%.3f/%.3f" % (auc, prec))
        print("  %-30s %s" % (p["name"][:30], "  ".join("%-14s" % x for x in row)))

    print("\n  %-30s %s" % ("平均 AUC / 保留60%准确率", ""))
    base = None
    for a in arms:
        m = np.array(acc[a]).mean(axis=0)
        if a == "cnn14":
            base = m
        print("  %-30s %.3f / %.3f" % (a, m[0], m[1]))
    if base is not None:
        print()
        for a in arms:
            if a == "cnn14":
                continue
            m = np.array(acc[a]).mean(axis=0)
            d = m[0] - base[0]
            print("  %s vs CNN14：AUC %+.3f，准确率 %+.3f  →  %s"
                  % (a, d, m[1] - base[1],
                     "更好" if d > 0.02 else ("持平" if abs(d) <= 0.02 else "更差")))




def end2end(data: Dict) -> None:
    """端到端：换骨干之后，**回合排序**变好还是变坏。

    候选级 AUC 和成片质量之间隔着「分组」这一层，而分组对候选的分布敏感。
    上一个实验（高分辨率 log-mel）就是候选级 0.812 > 0.794 赢了 CNN14，
    端到端 ρ 却是 0.246 vs 0.521 —— 等于没重排。所以这一步不能省。
    """
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from end2end_rerank import rank_rho
    from sklearn.linear_model import LogisticRegression
    from ppai import highlight

    cfg = config.load()
    hcfg = cfg["highlight"]
    per, arms = data["per"], data["arms"]
    y = np.concatenate([p["y"] for p in per])
    g = np.concatenate([np.full(len(p["y"]), p["name"]) for p in per])

    print("\n  端到端：回合长度排序与真值的 Spearman ρ\n")
    print("  %-30s %8s  %s" % ("留出", "不重排", "  ".join("%8s" % a for a in arms)))
    acc = {a: [] for a in arms}
    base_acc = []
    for p in per:
        if len(np.unique(p["y"])) < 2 or p.get("truth") is None:
            continue
        dur, truth, cand = p["dur"], p["truth"], p["cand"]
        b = rank_rho(truth, cand, dur, hcfg)["rho"]
        base_acc.append(b)
        row = []
        for a in arms:
            X = np.vstack([q["feats"][a] for q in per])
            tr, te = g != p["name"], g == p["name"]
            c = LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced")\
                .fit(X[tr], y[tr], sample_weight=_balance(g[tr]))
            pr = c.predict_proba(X[te])[:, 1]
            k = max(1, int(round(len(pr) * KEEP)))
            kept = np.sort(cand[np.argsort(-pr)[:k]])
            r = rank_rho(truth, kept, dur, hcfg)["rho"]
            acc[a].append(r)
            row.append("%8.3f" % r)
        print("  %-30s %8.3f  %s" % (p["name"][:30], b, "  ".join(row)))

    # **只在所有臂都有值的视频上取平均。** 用 nanmean 会让不同臂落在
    # 不同的子集上：cnn14 在 5cf65f 上有 0.618，另两个是 NaN，
    # 结果 cnn14 的均值多了一个偏高的样本 —— 这就是幸存者偏差，
    # 这个项目在 gap 调参时已经栽过一次（见 README）。
    ok = [i for i in range(len(base_acc))
          if not np.isnan(base_acc[i]) and all(not np.isnan(acc[a][i]) for a in arms)]
    dropped = len(base_acc) - len(ok)
    print("\n  共同可比的视频 %d 个（%d 个因某臂算不出 ρ 被整体排除）" % (len(ok), dropped))
    print("  平均 ρ    不重排 %.3f" % np.mean([base_acc[i] for i in ok]))
    base = None
    for a in arms:
        m = float(np.mean([acc[a][i] for i in ok]))
        if a == "cnn14":
            base = m
        print("            %-10s %.3f" % (a, m))
    if base is not None:
        print()
        for a in arms:
            if a == "cnn14":
                continue
            d = float(np.mean([acc[a][i] for i in ok])) - base
            print("  %s vs CNN14：ρ %+.3f  →  %s" % (
                a, d, "更好，可以换" if d > 0.03 else
                ("持平 —— 那就选小的那个" if abs(d) <= 0.03 else "**更差，不能换**")))


if __name__ == "__main__":
    arms = sys.argv[1:] or ["cnn14", "mn04_as", "mn10_as"]
    print("对照臂：%s\n" % ", ".join(arms))
    d = collect(arms)
    lovo(d)
    end2end(d)

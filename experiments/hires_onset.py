"""实验：用高时间分辨率的 log-mel 补丁替掉 1 秒的 PANNs 向量。

动机（已测量，不是猜的）
------------------------
PANNs CNN14 的窗口是 1 秒 / 跳 0.5 秒，而我们要分辨的事件是 5-15 毫秒的瞬态，
发球四拍之间只隔 0.143 / 0.290 / 0.223 秒 —— 一个窗口里装着 3-7 个事件。
所以候选处取到的 2048 维向量描述的是「周围一秒的声学环境」而不是「这一下」。
两个失败都指向这一点：
  * 挥拍/弹跳分类器 AUC 最高 0.730，端到端 38% 反而不如基线 48%
  * 回合级上 PANNs 0.773，打不过谱通量基线 0.793

三个对照臂
----------
  A  PANNs 2048 维 + 逻辑回归      —— 现役基线
  B  log-mel 补丁拉平 + 逻辑回归    —— **关键对照**
  C  log-mel 补丁 + 小 CNN

B 是这个实验的重点。A 和 B 的模型完全一样（同一个逻辑回归、同样的正则、
同样的按视频均衡权重），维度也接近（2048 vs 1984），**唯一的差别是特征的
时间分辨率**。B 若赢 A，说明瓶颈是分辨率；B 若不赢，那 C 赢了也只能说明
是模型容量的功劳，和「高分辨率」这个假设无关。

评测协议和 rerank.train 一致：留一视频交叉验证，按比例保留 60% 后看准确率。
不用固定概率阈值 —— 概率标定不跨视频（见 rerank.py 开头）。
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppai import audio, config, labels as L, rerank  # noqa: E402

# 10 毫秒一帧 —— 和检测器用的是同一套 STFT 参数，补丁能直接对齐到候选帧
PATCH_HALF = 15          # ±15 帧 = ±150 毫秒
N_MELS = 64
KEEP = 0.6               # 与线上一致：按比例保留 60%


def mel_filters(sr: int, n_fft: int, n_mels: int,
                fmin: float = 60.0, fmax: float = None) -> np.ndarray:
    """三角 mel 滤波器组。自己写而不是引 torchaudio ——
    这套参数将来要原样移植到 Kotlin，写在这里省得两边对不上。"""
    fmax = fmax or sr / 2.0
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l: c = l + 1
        if r == c: r = c + 1
        r = min(r, fb.shape[1] - 1); c = min(c, r - 1); l = min(l, c - 1)
        if l < 0 or c <= l or r <= c: continue
        fb[i, l:c] = (np.arange(l, c) - l) / float(c - l)
        fb[i, c:r] = (r - np.arange(c, r)) / float(r - c)
    return fb


def logmel(x: np.ndarray, cfg: Dict) -> np.ndarray:
    """[n_mels, n_frames]。和 onset_envelope 用同样的窗和跳，帧号可直接互换。"""
    sr, n_fft, hop = cfg["sr"], cfg["n_fft"], cfg["hop"]
    win = np.hanning(n_fft).astype(np.float32)
    if len(x) < n_fft:
        return np.zeros((N_MELS, 0), dtype=np.float32)
    n = 1 + (len(x) - n_fft) // hop
    fb = mel_filters(sr, n_fft, N_MELS)
    out = np.zeros((N_MELS, n), dtype=np.float32)
    off = np.arange(n_fft)
    for s in range(0, n, 2000):
        e = min(s + 2000, n)
        idx = (np.arange(s, e) * hop)[:, None] + off[None, :]
        mag = np.abs(np.fft.rfft(x[idx] * win, axis=1)).astype(np.float32)
        out[:, s:e] = np.log1p((mag @ fb.T) * 100.0).T
    return out


def patches(lm: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """以候选帧为中心切补丁，边界补零。返回 [N, n_mels, 2*half+1]。"""
    w = 2 * PATCH_HALF + 1
    out = np.zeros((len(frames), lm.shape[0], w), dtype=np.float32)
    for i, f in enumerate(frames):
        a, b = f - PATCH_HALF, f + PATCH_HALF + 1
        sa, sb = max(0, a), min(lm.shape[1], b)
        if sb > sa:
            out[i, :, sa - a: sa - a + (sb - sa)] = lm[:, sa:sb]
    return out


def build() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (PANNs 特征, log-mel 补丁, 标签, 分组)。两种特征来自**同一批候选**。"""
    cfg = config.load()
    acfg = cfg["audio"]
    Xp, Xm, y, g = [], [], [], []
    for path in sorted(glob.glob("labels/*.json")):
        lab = L.load(path)
        rng = L.scored_ranges(lab)
        neg = lab.get("negative_ranges") or []
        if not rng and not neg:
            continue
        if rng and not lab["playing"]:
            continue
        video = lab["video"]
        if not os.path.exists(video):
            print("  跳过（原片不在，算不了 log-mel）: %s" % os.path.basename(path))
            continue

        pcm = audio.extract_pcm(video, acfg["sr"])
        det, env, _, fr = audio.detect_hits(pcm, acfg)
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
        yy = np.array([0 if in_neg[np.searchsorted(det, c)]
                       else (1 if len(truth) and np.any(np.abs(truth - c) <= rerank.TOL) else 0)
                       for c in cand])

        lm = logmel(pcm, acfg)
        frames = np.clip((cand * fr).round().astype(int), 0, max(lm.shape[1] - 1, 0))
        Xm.append(patches(lm, frames))
        Xp.append(rerank._features(video, cand, "cache"))
        y.append(yy)
        g.append(np.full(len(cand), os.path.basename(path)))
        print("  %-38s %4d 候选 / %3d 真 (准确率 %.3f)"
              % (os.path.basename(path)[:38], len(cand), yy.sum(), yy.mean()))
    return (np.vstack(Xp), np.concatenate(Xm), np.concatenate(y), np.concatenate(g))


def _balance(groups: np.ndarray) -> np.ndarray:
    w = np.ones(len(groups), float)
    for s in set(groups):
        m = groups == s
        w[m] = len(groups) / (len(set(groups)) * m.sum())
    return w


def eval_lr(Xtr, ytr, gtr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    c = LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced")\
        .fit(Xtr, ytr, sample_weight=_balance(gtr))
    p = c.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), prec_at_keep(p, yte)


def prec_at_keep(p, yte):
    k = max(1, int(len(p) * KEEP))
    return float(yte[np.argsort(-p)[:k]].mean())


def eval_cnn(Xtr, ytr, gtr, Xte, yte, seed=0):
    """小 CNN。**故意做小** —— 训练集只有几百个样本，容量给大了就是在背答案。"""
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(48, 1))
    n_par = sum(p.numel() for p in net.parameters())

    mu, sd = Xtr.mean(), Xtr.std() + 1e-6
    xt = torch.from_numpy(((Xtr - mu) / sd)[:, None]).float()
    yt = torch.from_numpy(ytr).float()[:, None]
    wt = torch.from_numpy(_balance(gtr) *
                          np.where(ytr == 1, len(ytr) / (2 * max(ytr.sum(), 1)),
                                   len(ytr) / (2 * max((1 - ytr).sum(), 1)))).float()[:, None]
    xe = torch.from_numpy(((Xte - mu) / sd)[:, None]).float()

    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    n = len(xt)
    for ep in range(60):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            l = (lossf(net(xt[b]), yt[b]) * wt[b]).mean()
            l.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(xe)).numpy().ravel()
    return roc_auc_score(yte, p), prec_at_keep(p, yte), n_par


def main():
    print("构建数据集（两种特征来自同一批候选）\n")
    Xp, Xm, y, g = build()
    files = sorted(set(g))
    print("\n合计 %d 候选 / %d 真，来自 %d 份标注" % (len(y), y.sum(), len(files)))
    print("PANNs 维度 %d   log-mel 补丁 %s（拉平 %d）\n"
          % (Xp.shape[1], Xm.shape[1:], Xm[0].size))

    Xf = Xm.reshape(len(Xm), -1)
    rows = []
    for f in files:
        tr, te = g != f, g == f
        if len(np.unique(y[te])) < 2:
            print("  留出 %-34s 只有一类，跳过" % f[:34])
            continue
        a_auc, a_p = eval_lr(Xp[tr], y[tr], g[tr], Xp[te], y[te])
        b_auc, b_p = eval_lr(Xf[tr], y[tr], g[tr], Xf[te], y[te])
        c_auc, c_p, npar = eval_cnn(Xm[tr], y[tr], g[tr], Xm[te], y[te])
        base = float(y[te].mean())
        rows.append((f, base, a_auc, a_p, b_auc, b_p, c_auc, c_p))
        print("  留出 %-30s 基线 %.3f | A %.3f/%.3f | B %.3f/%.3f | C %.3f/%.3f"
              % (f[:30], base, a_auc, a_p, b_auc, b_p, c_auc, c_p))

    if rows:
        m = np.array([r[2:] for r in rows], float).mean(axis=0)
        b = np.mean([r[1] for r in rows])
        print("\n  %-34s %s" % ("", "AUC     保留60%准确率"))
        print("  %-34s %.3f   %.3f  （不重排的基线准确率 %.3f）" % ("A  PANNs 1秒窗 + 逻辑回归", m[0], m[1], b))
        print("  %-34s %.3f   %.3f" % ("B  log-mel 10ms 补丁 + 逻辑回归", m[2], m[3]))
        print("  %-34s %.3f   %.3f  （%d 参数）" % ("C  log-mel 补丁 + 小 CNN", m[4], m[5], npar))
        print()
        if m[2] > m[0] + 0.02:
            print("  → B 赢过 A：瓶颈确实是**时间分辨率**，不是模型容量。")
        elif m[4] > m[0] + 0.02:
            print("  → B 没赢但 C 赢了：说明是**模型容量**的功劳，"
                  "「高分辨率」这个假设没被证实。")
        else:
            print("  → **两个都没赢过 PANNs。** 高分辨率补丁这条路在当前数据量下不成立。")


if __name__ == "__main__":
    main()

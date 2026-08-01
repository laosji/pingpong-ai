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

from .paths import MODELS

MODEL_PATH = os.path.join(MODELS, "rerank.pkl")
TOL = 0.35        # 候选与人工标记的匹配容差，与 evaluate.hit_level 保持一致


def _features(video: str, cand: np.ndarray, cache_dir: Optional[str]) -> np.ndarray:
    from . import embed
    t, f = embed.embed_video(video, cache_dir=cache_dir)
    if len(t) == 0:
        return np.zeros((len(cand), 0))
    idx = np.array([np.argmin(np.abs(t - c)) for c in cand])
    return f[idx]


def freeze(video: str, label_path: str, cfg: Dict,
           cache_dir: Optional[str] = None) -> str:
    """把标注区间内的候选时刻和嵌入固化到 .npz，让标注脱离原视频独立存在。

    **这是「视频临时存储、到期自动删除」架构的前提。** 训练需要重读原片
    算嵌入（见 build_dataset），视频一删，该标注就在训练时被静默跳过 ——
    不报错，只是某天发现指标不再涨。用户反馈这条复利机制会就这么废掉。

    只固化标注区间内的候选，不是整个视频：一次反馈通常只涉及几十个候选，
    2048 维 float16 约 100KB，相比原片几十上百 MB 可以忽略。
    """
    from . import audio, labels as L

    lab = L.load(label_path)
    rng = list(L.scored_ranges(lab)) + list(lab.get("negative_ranges") or [])
    if not rng:
        return ""
    pcm = audio.extract_pcm(video, cfg["audio"]["sr"])
    det, _, _, _ = audio.detect_hits(pcm, cfg["audio"])
    keep = np.zeros(len(det), bool)
    for a, b in rng:
        keep |= (det >= a) & (det <= b)
    cand = det[keep]
    if len(cand) == 0:
        return ""
    X = _features(video, cand, cache_dir)
    dst = os.path.splitext(label_path)[0] + ".npz"
    np.savez_compressed(dst, times=cand, feats=X.astype(np.float16))
    return dst


def _frozen(label_path: str):
    p = os.path.splitext(label_path)[0] + ".npz"
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d["times"], d["feats"].astype(np.float32)


def build_dataset(label_dir: str, cfg: Dict,
                  cache_dir: Optional[str] = None) -> Tuple:
    """从所有带 complete_ranges 或 negative_ranges 的标注里构造训练集。

    优先用固化的 .npz —— 原视频可能已按生命周期规则删除。
    """
    from . import audio, labels as L

    X, y, groups = [], [], []
    for path in sorted(glob.glob(os.path.join(label_dir, "*.json"))):
        lab = L.load(path)
        rng = L.scored_ranges(lab)
        neg = lab.get("negative_ranges") or []
        # 用户在产品里点「这段不对」产生的区间：没有击球标注也能用 ——
        # 那是用户明确声称「这里没有有效击球」，区间内所有候选都是可信负例。
        # 这是数据闭环的入口：用户越用，负例越多，不需要人工逐拍标。
        if not rng and not neg:
            continue
        if rng and not lab["playing"]:
            continue
        video = lab["video"]
        frz = _frozen(path)
        if not os.path.exists(video) and frz is None:
            print("  跳过（视频和固化嵌入都不在）: %s" % os.path.basename(video))
            continue
        # 上线前的体检。原来只有下面那一条「零击球 + 有捡球」的守卫，
        # 它挡不住「标到一半换错轨道」：实测碰到过一份前 19 秒 playing 正常、
        # 之后 99 段击球全标进 pickup 的，守卫看到前面有 playing 就放行，
        # 结果 424 个真击球被当成负例喂进模型。比不标这份还糟。
        # audit() 用形状、发球位置、声明与实际是否相符三条判据，
        # **拦下来而不是静默使用**。
        problems = [p for p in L.audit(lab) if p["level"] == "block"]
        if problems:
            print("  ⛔ 跳过 %s：" % os.path.basename(path))
            for p in problems:
                print("       %s" % p["what"])
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

        if frz is not None:
            det, feats = frz
        else:
            pcm = audio.extract_pcm(video, cfg["audio"]["sr"])
            det, _, _, _ = audio.detect_hits(pcm, cfg["audio"])
            feats = None
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
                       else (1 if len(truth) and np.any(np.abs(truth - c) <= TOL) else 0)
                       for c in cand])
        X.append(feats[keep] if feats is not None else _features(video, cand, cache_dir))
        y.append(yy)
        groups.append(np.full(len(cand), os.path.basename(path)))
        tag = "" if not neg else "  [含 %d 段用户反馈]" % len(neg)
        if frz is not None:
            tag += "  [用固化嵌入，无需原片]"
        print("  %-40s %4d 候选 / %3d 真 (准确率 %.3f)%s"
              % (os.path.basename(video)[:40], len(cand), yy.sum(), yy.mean(), tag))
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
          cache_dir: Optional[str] = None) -> Dict:
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


class _Logit:
    """逻辑回归的推理部分，纯 numpy。

    存在的理由只有一个：**桌面 app 里不该为了 predict_proba 装 47MB 的
    scikit-learn**（它还会拖来 98MB 的 scipy）。这个模型就是 2048 维的
    LogisticRegression，推理是一次点积加一个 sigmoid，三行 numpy 完事。

    sklearn 仍然是训练时的依赖 —— train() 那边不动。
    """

    def __init__(self, coef: np.ndarray, intercept: np.ndarray):
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(1, -1)
        self.intercept_ = np.asarray(intercept, dtype=np.float64).reshape(1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(X, dtype=np.float64) @ self.coef_.ravel() + self.intercept_[0]
        # 先裁再取指数，否则大负值那侧 exp 会溢出（只是 warning，但结果会是 nan）
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
        return np.column_stack([1.0 - p, p])


def _npz_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".npz"


def load(path: str = MODEL_PATH):
    """优先读 .npz（只依赖 numpy），没有再退回 .pkl（要 sklearn）。

    打包 app 时只带 .npz，所以那边永远走第一条。
    """
    npz = _npz_path(path)
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=False)
        return {"clf": _Logit(d["coef"], d["intercept"]),
                "n_train": int(d["n_train"]) if "n_train" in d else 0,
                "files": [str(x) for x in d["files"]] if "files" in d else []}
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def export_npz(path: str = MODEL_PATH) -> str:
    """把 sklearn 的 .pkl 转成只依赖 numpy 的 .npz。训练后跑一次。"""
    with open(path, "rb") as fh:
        m = pickle.load(fh)
    dst = _npz_path(path)
    np.savez(dst, coef=m["clf"].coef_, intercept=m["clf"].intercept_,
             n_train=m.get("n_train", 0),
             files=np.array(m.get("files", []), dtype=object).astype(str))
    return dst


SAMPLE_AT = (0.2, 0.5, 0.8)   # 抽样位置（占全片比例）
SAMPLE_WIN = 40               # 每段秒数


def _sample_audio(video: str, dst_dir: str, cfg: Dict) -> str:
    """从视频里抽三段各 40 秒的音频拼成一个 wav。太短的直接用原片。"""
    import subprocess
    from .media import probe

    dur = probe(video)["duration"]
    if dur <= len(SAMPLE_AT) * SAMPLE_WIN:
        return video
    parts = []
    for i, frac in enumerate(SAMPLE_AT):
        p = os.path.join(dst_dir, "s%d.wav" % i)
        subprocess.run(["ffmpeg", "-y", "-v", "quiet",
                        "-ss", str(dur * frac), "-t", str(SAMPLE_WIN),
                        "-i", video, "-vn", "-ac", "1",
                        "-ar", str(cfg["audio"]["sr"]), p], check=True)
        parts.append(p)
    lst = os.path.join(dst_dir, "list.txt")
    with open(lst, "w") as fh:
        fh.write("".join("file '%s'\n" % p for p in parts))
    out = os.path.join(dst_dir, "sample.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", out], check=True)
    return out


def looks_like_pingpong(video: str, cfg: Dict, model=None) -> Dict:
    """判断这段素材是不是乒乓球录像。返回 {"ok": bool, "score": float}。

    **用重排器的平均概率，不用场景检测。** 场景检测那条（scene.assess 的
    中下区域水平边缘）在阴性对照上直接失效：neg_speech 的 edge 是 0.255，
    比所有真实素材（0.141-0.213）都高 —— 说话视频里的桌子书架比球台边缘
    还强。音频的瞬态密度也分不开（阴性 1.08-1.27/秒 vs 真实 0.70-2.19/秒），
    说话的爆破音在密度上和击球一模一样。

    重排器是唯一分得开的。它在真实击球上训过，全片实测：

        阴性对照   均值 0.009 - 0.010
        真实素材   均值 0.379 - 0.672      差 40 倍

    这里用**平均**概率不违反本模块开头「概率标定不跨视频」那条 —— 那条说的是
    别拿固定阈值挑单个候选（会把召回压到 0.170）。这里只是拿全片均值做一次
    粗粒度的领域判断，两边差 40 倍，标定漂移那点量级不影响。

    **抽样而不是全片**：三段各 40 秒（20%/50%/80% 处）。全片跑要花时长的
    4.2%，一小时素材就是 2.5 分钟，上传请求挂不住；抽样固定 3.5 秒，
    且实测分离度没退化（真实 0.415-0.700 / 阴性 0.009-0.010）。
    抽样音频不写进 cache —— 它不是原片的嵌入，混进去会污染剪辑。

    模型没训练时返回 ok=True：不能因为自己没模型就把人挡在门外。
    """
    import tempfile
    from . import audio

    model = model or load(cfg.get("rerank", {}).get("model", MODEL_PATH))
    if model is None:
        return {"ok": True, "score": None}
    gate = float(cfg.get("rerank", {}).get("pingpong_gate", 0.10))
    with tempfile.TemporaryDirectory() as tmp:
        src = _sample_audio(video, tmp, cfg)
        pcm = audio.extract_pcm(src, cfg["audio"]["sr"])
        det, _, _, _ = audio.detect_hits(pcm, cfg["audio"])
        if len(det) < 10:
            # 两分钟里十个瞬态都没有 —— 没声音或者没在打球，剪不出东西
            return {"ok": False, "score": 0.0}
        X = _features(src, det, tmp)
    if X.shape[1] == 0:
        return {"ok": True, "score": None}
    score = float(model["clf"].predict_proba(X)[:, 1].mean())
    return {"ok": score >= gate, "score": score}


def apply(video: str, cand: np.ndarray, model, keep_ratio: float = 0.6,
          cache_dir: Optional[str] = None) -> np.ndarray:
    """返回保留下来的候选（按比例，不用固定概率阈值 —— 概率标定不跨视频）。"""
    if model is None or len(cand) == 0:
        return cand
    X = _features(video, cand, cache_dir)
    if X.shape[1] == 0:
        return cand
    p = model["clf"].predict_proba(X)[:, 1]
    k = max(1, int(round(len(cand) * keep_ratio)))
    # **并列必须有确定的打破规则。** 嵌入窗是 1 秒 / 跳 0.5 秒，所以间隔小于
    # 0.5 秒的两个候选会落在同一个窗上，特征完全相同、概率**精确相等**
    # （实测 5.52 秒和 5.70 秒的概率差是 0.000e+00）。
    # np.argsort 默认快排、不稳定，同样的输入在不同 numpy 版本、
    # 不同数组长度下可能给出不同的胜者，而谁胜出会改变回合边界。
    # kind="stable" 让并列时按下标升序（= 时间升序），
    # Kotlin 侧用 thenBy{index} 是同一个规则。
    return np.sort(cand[np.argsort(-p, kind="stable")[:k]])

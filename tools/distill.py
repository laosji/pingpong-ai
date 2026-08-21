"""把 323MB 的 CNN14 + 逻辑回归蒸成一个 1.8MB 的小网。

用法
----
    python3 tools/distill.py data   --n 120000     # 生成蒸馏集（教师打标）
    python3 tools/distill.py train  --epochs 30
    python3 tools/distill.py export
    python3 tools/distill.py eval                  # **端到端**对比，不是看 AUC

为什么不存波形
--------------
一条样本是 32000 个 float。10 万条就是 12GB。所以只存**配方**
（哪段素材、从哪一帧起、随机种子），训练时按种子确定性地重建 ——
增广全是 numpy 的确定性运算，同一个种子必然得到同一段波形。
元数据一条 12 字节，10 万条 1.2MB。

为什么先增广再让教师打标
------------------------
反过来做（教师标原始窗口，然后增广波形沿用原标签）是错的：
增大三倍音量、混进说话声之后，教师给的概率本来就变了，
沿用旧标签等于教学生忽略这些变化。
先增广、再让教师看**增广后的信号**，学生学到的才是教师这个函数本身。

素材只有 11 分钟，靠什么撑住
----------------------------
一是增广（增益/噪声/极性/变速/混音）；二是**故意喂大量「显然不是乒乓球」
的输入**（纯噪声、正弦、静音、合成素材），教师会把它们标成接近 0。
门禁要的正是这个能力，而这类输入不需要真实素材就能造出无限多。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppai import audio as A                                    # noqa: E402
from ppai.student import PANNS_SR, WIN_SAMPLES                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "distill")


def sources() -> list:
    """所有能拿到的音频。**阴性和合成素材同样重要** ——
    学生要学的不只是「什么是击球」，还有「什么不是」。"""
    data = os.environ.get("PIPO_DATA_DIR",
                          os.path.expanduser("~/Library/Application Support/Pipo"))
    cands = []
    up = os.path.join(data, "uploads")
    for root, _, files in os.walk(up):
        for f in files:
            if f.lower().endswith((".mp4", ".mov", ".m4v")):
                cands.append(os.path.join(root, f))
    for f in ("negatives/neg_speech.mp4", "negatives/neg_misc.mp4",
              "android/cutter/src/androidTest/assets/land.mp4",
              "android/cutter/src/androidTest/assets/port.mp4"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            cands.append(p)
    return cands


def load_all(paths: list) -> list:
    out = []
    for p in paths:
        pcm = A.extract_pcm(p, PANNS_SR)
        if len(pcm) >= WIN_SAMPLES:
            out.append(pcm.astype(np.float32))
            print("  %7.1f 秒  %s" % (len(pcm) / PANNS_SR, os.path.basename(p)))
    return out


# ---------------------------------------------------------------- 增广

def render(tracks: list, src: int, start: int, seed: int) -> np.ndarray:
    """按配方重建一个 1 秒窗。**必须是种子的纯函数** —— 训练时要能重现。

    src < 0 表示合成输入（噪声/正弦/静音），不取自任何素材。
    """
    rng = np.random.default_rng(seed)
    if src >= 0:
        x = tracks[src][start:start + WIN_SAMPLES].copy()
        if len(x) < WIN_SAMPLES:
            x = np.pad(x, (0, WIN_SAMPLES - len(x)))
    else:
        kind = -src
        t = np.arange(WIN_SAMPLES) / PANNS_SR
        if kind == 1:                                  # 白噪声
            x = rng.standard_normal(WIN_SAMPLES).astype(np.float32) * rng.uniform(0.005, 0.3)
        elif kind == 2:                                # 粉噪声（低频更强，像空调/嗡鸣）
            w = rng.standard_normal(WIN_SAMPLES)
            f = np.fft.rfft(w)
            k = np.arange(len(f)); k[0] = 1
            x = np.fft.irfft(f / np.sqrt(k), WIN_SAMPLES).astype(np.float32)
            x *= rng.uniform(0.01, 0.4) / (np.abs(x).max() + 1e-9)
        elif kind == 3:                                # 纯音 + 谐波（哨声、蜂鸣）
            f0 = rng.uniform(200, 4000)
            x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3)).astype(np.float32)
            x *= rng.uniform(0.02, 0.5)
        elif kind == 4:                                # 近乎静音
            x = rng.standard_normal(WIN_SAMPLES).astype(np.float32) * rng.uniform(1e-5, 2e-3)
        else:
            # **非乒乓球的瞬态。这一类不能少。**
            # 检测器本来就是靠宽带瞬态触发的，重排器的全部工作就是分辨
            # 「乒乓球击球」和「别的咔哒声」。学生要是从没见过后者，
            # 就会学成「有瞬态就是击球」，门禁直接失效。
            # 频带、时长、衰减、疏密全随机，故意覆盖到击球的邻域又不等于它。
            x = rng.standard_normal(WIN_SAMPLES).astype(np.float32) * rng.uniform(1e-4, 5e-3)
            for _ in range(int(rng.integers(1, 14))):
                at = int(rng.integers(0, WIN_SAMPLES - 800))
                ln = int(rng.integers(40, 700))
                lo, hi = sorted(rng.uniform(150, 15000, 2))
                if hi - lo < 300:
                    hi = lo + 300
                w = rng.standard_normal(ln)
                F = np.fft.rfft(w)
                fr_ = np.fft.rfftfreq(ln, 1.0 / PANNS_SR)
                F[(fr_ < lo) | (fr_ > hi)] = 0
                b = np.fft.irfft(F, ln).astype(np.float32)
                b *= np.exp(-np.linspace(0, rng.uniform(2, 12), ln)).astype(np.float32)
                mx = np.abs(b).max()
                if mx > 0:
                    x[at:at + ln] += b / mx * rng.uniform(0.05, 0.9)
        return np.clip(x, -1, 1).astype(np.float32)

    # --- 增广链。每一项都可能不生效，让分布覆盖「原样」到「面目全非」---
    if rng.random() < 0.5:                              # 变速（重采样后裁/补）
        r = rng.uniform(0.93, 1.07)
        n = int(WIN_SAMPLES / r)
        idx = np.clip((np.arange(n) * r).astype(np.int64), 0, WIN_SAMPLES - 1)
        y = x[idx]
        x = np.pad(y, (0, WIN_SAMPLES - len(y)))[:WIN_SAMPLES] if len(y) < WIN_SAMPLES \
            else y[:WIN_SAMPLES]
    if rng.random() < 0.25:                             # 混进另一段素材（隔壁球台 / 说话）
        j = int(rng.integers(0, len(tracks)))
        if len(tracks[j]) > WIN_SAMPLES:
            s2 = int(rng.integers(0, len(tracks[j]) - WIN_SAMPLES))
            x = x + tracks[j][s2:s2 + WIN_SAMPLES] * rng.uniform(0.1, 0.5)
    if rng.random() < 0.6:                              # 底噪
        x = x + rng.standard_normal(WIN_SAMPLES).astype(np.float32) * rng.uniform(1e-4, 0.02)
    if rng.random() < 0.25:                             # 一阶滤波（远近 / 麦克风差异）
        a = rng.uniform(-0.7, 0.7)
        x = np.concatenate([x[:1], x[1:] - a * x[:-1]]).astype(np.float32)
    x = x * rng.uniform(0.2, 3.0)                       # 增益
    if rng.random() < 0.5:
        x = -x                                          # 极性
    return np.clip(x, -1, 1).astype(np.float32)


def make_recipes(tracks: list, n: int, weights: np.ndarray,
                 seed: int = 0) -> np.ndarray:
    """[n, 3] 的 int64：(src, start, seed)。四分之一是合成输入。

    **素材不能均匀取。** 第一版是 `randint(0, len(tracks))`，于是 443 秒的
    真实对打和 10 秒的合成素材被抽到的机会一样多，正类被稀释到 4% ——
    学生只要一律输出接近 0 就能拿到很低的损失，等于什么都没学。
    [weights] 按时长加权，并给真实对打额外配额。
    """
    rng = np.random.default_rng(seed)
    rows = np.zeros((n, 3), dtype=np.int64)
    n_syn = n // 4
    w = weights / weights.sum()
    picks = rng.choice(len(tracks), size=n - n_syn, p=w)
    for i in range(n):
        if i < n_syn:
            rows[i] = (-int(rng.integers(1, 6)), 0, int(rng.integers(1, 2**31)))
        else:
            s = int(picks[i - n_syn])
            hi = max(1, len(tracks[s]) - WIN_SAMPLES)
            rows[i] = (s, int(rng.integers(0, hi)), int(rng.integers(1, 2**31)))
    rng.shuffle(rows)
    return rows


def source_weights(paths: list, tracks: list) -> np.ndarray:
    """按时长加权，真实对打再乘 3。

    真实素材只有 443 秒，是整件事里最稀缺的东西；阴性和合成输入
    要多少有多少。不加权的话正类样本会被淹掉。
    """
    w = np.array([len(t) for t in tracks], dtype=np.float64)
    for i, p in enumerate(paths[:len(tracks)]):
        b = os.path.basename(p)
        if "neg_" not in b and b not in ("land.mp4", "port.mp4"):
            w[i] *= 3.0
    return w


# ---------------------------------------------------------------- 各子命令

def cmd_data(args):
    from ppai import embed, rerank
    os.makedirs(OUT, exist_ok=True)
    paths = sources()
    print("素材：")
    tracks = load_all(paths)
    if not tracks:
        raise SystemExit("一段音频都没找到")

    rows = make_recipes(tracks, args.n, source_weights(paths, tracks), seed=args.seed)
    sess = embed._load_onnx()
    head = rerank.load(rerank.MODEL_PATH)

    probs = np.zeros(len(rows), dtype=np.float32)
    B = 64
    t0 = time.time()
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        xb = np.stack([render(tracks, int(s), int(st), int(sd)) for s, st, sd in chunk])
        f = sess.run(["embedding"], {"pcm": xb.astype(np.float32)})[0]
        probs[i:i + len(chunk)] = head["clf"].predict_proba(f)[:, 1]
        if (i // B) % 20 == 0:
            done = i + len(chunk)
            el = time.time() - t0
            eta = el / max(done, 1) * (len(rows) - done)
            print("  %6d / %d   %.0f 窗/秒   剩 %.0f 分钟"
                  % (done, len(rows), done / max(el, 1e-9), eta / 60), flush=True)

    np.save(os.path.join(OUT, "recipes.npy"), rows)
    np.save(os.path.join(OUT, "targets.npy"), probs)
    with open(os.path.join(OUT, "sources.json"), "w") as f:
        json.dump([os.path.basename(p) for p in paths], f, ensure_ascii=False)
    hi = float((probs > 0.5).mean())
    print("\n  %d 条，正类占比 %.1f%%，均值 %.4f" % (len(rows), 100 * hi, probs.mean()))
    print("  存到 %s" % OUT)


def _load_set():
    rows = np.load(os.path.join(OUT, "recipes.npy"))
    tgt = np.load(os.path.join(OUT, "targets.npy"))
    return rows, tgt


class _Set:
    """按配方即时重建波形。**不缓存** —— 缓存 12 万条要 12GB，
    而重建一条只要不到 1 毫秒，比等磁盘还快。"""

    def __init__(self, tracks, rows, tgt):
        self.tracks, self.rows, self.tgt = tracks, rows, tgt

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        s, st, sd = (int(v) for v in self.rows[i])
        x = render(self.tracks, s, st, sd)
        return torch.from_numpy(x), torch.tensor(float(self.tgt[i]))


def cmd_train(args):
    import torch
    from torch.utils.data import DataLoader
    from ppai.student import Student, n_params

    torch.manual_seed(0)
    paths = sources()
    tracks = load_all(paths)
    rows, tgt = _load_set()

    # 划分按**配方下标**，不按素材 —— 素材只有五段，按素材划会让验证集
    # 要么全是阴性要么全是阳性。这里验的是「学得像不像教师」，
    # 泛化到新场馆是另一回事，靠端到端那一步（cmd_eval）看。
    n_val = max(1, len(rows) // 10)
    idx = np.random.default_rng(0).permutation(len(rows))
    vi, ti = idx[:n_val], idx[n_val:]

    # **worker 必须常驻。** macOS 上 DataLoader 用 spawn，每建一次 worker 就要
    # 把 tracks（86MB 的音频）pickle 过去一遍；默认是**每轮重建**，25 轮就是
    # 25 次这个开销，而且实测会卡死（4 个 worker 全退出、主进程 0% CPU 干等）。
    # persistent_workers 让它们建一次用到底。
    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    if args.workers > 0:
        kw["prefetch_factor"] = 4        # 渲染和 MPS 计算重叠起来
    tr = DataLoader(_Set(tracks, rows[ti], tgt[ti]), batch_size=args.batch,
                    shuffle=True, drop_last=True, **kw)
    va = DataLoader(_Set(tracks, rows[vi], tgt[vi]), batch_size=args.batch, **kw)

    # STFT 前端是 conv1d（514 个通道、核长 512），在 CPU 上一轮要 8 分钟，
    # 在 MPS 上 1 分钟。用 rfft 会更省，但那样导出的 ONNX 图和训练图就不是
    # 同一个，得再验一次一致性 —— 不值得，换个设备就解决了。
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = Student(width=args.width).to(dev)
    print("  学生 %d 个可训练参数，训练设备 %s" % (n_params(m), dev), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tr))

    best = 1e9
    for ep in range(args.epochs):
        t_ep = time.time()
        m.train()
        run = 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            # **软目标上的 BCE，不是 MSE。** 教师给的是概率，
            # BCE 在 logit 空间的梯度对两端（接近 0 / 接近 1）更敏感，
            # 而这两端恰好是我们最在乎的：门禁看均值、重排看排序。
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x), y)
            loss.backward()
            opt.step(); sched.step()
            run += loss.item() * len(x)
        m.eval()
        vs, vn, err = 0.0, 0, 0.0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                p = torch.sigmoid(m(x))
                vs += torch.nn.functional.binary_cross_entropy(p, y).item() * len(x)
                err += (p - y).abs().sum().item()
                vn += len(x)
        print("  第 %2d 轮  训练 %.4f  验证 %.4f  平均概率误差 %.4f  用时 %.0f 秒"
              % (ep + 1, run / len(tr.dataset), vs / vn, err / vn,
                 time.time() - t_ep), flush=True)
        if vs / vn < best:
            best = vs / vn
            torch.save({k: v.cpu() for k, v in m.state_dict().items()},
                       os.path.join(OUT, args.out))
    print("  最好验证损失 %.4f，权重存到 distill/%s" % (best, args.out))


def cmd_export(args):
    import struct

    import torch
    from ppai.student import Student

    class Wrap(torch.nn.Module):
        """导出时把 (B,) 变成 (B,1)。

        **这是为了 Kotlin 一行都不用改。** 现有代码把 ONNX 输出读成
        `Array<FloatArray>`（[B, DIM]），再拿 rerank.bin 的系数算
        `sigmoid(f·coef + b)`。学生输出 [B,1]、配一个 coef=[1.0]、b=0 的
        rerank.bin，算出来正好就是 sigmoid(logit) —— 同一条代码路径，
        同一个阈值口径，门禁的 0.10 和重排的 0.6 都不用重新标定。
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x).unsqueeze(-1)

    m = Student(width=args.width)
    m.load_state_dict(torch.load(os.path.join(OUT, args.src), map_location="cpu"))
    m.eval()
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    dst = os.path.join(ROOT, "models", "student.onnx")
    # 输入签名和 CNN14 完全一致：(n, 32000) float32
    torch.onnx.export(
        Wrap(m).eval(), torch.randn(2, WIN_SAMPLES), dst,
        input_names=["pcm"], output_names=["embedding"],
        dynamic_axes={"pcm": {0: "n"}, "embedding": {0: "n"}},
        opset_version=17, dynamo=False)

    # 配套的「恒等」权重文件，格式和 rerank.bin 一样（见 Rerank.kt 的 loadWeights）
    wb = os.path.join(ROOT, "models", "rerank_student.bin")
    with open(wb, "wb") as f:
        # 格式见 Rerank.kt 的 readWeights：魔数 + 版本 + 维度 + 截距 + 系数
        f.write(b"PIPO")
        f.write(struct.pack("<Iif", 1, 1, 0.0))
        f.write(struct.pack("<f", 1.0))             # coef[0] = 1 -> probability() 就是 sigmoid(logit)
    mb = os.path.getsize(dst) / 1e6
    print("  导出 %s  %.2f MB（教师 323.0 MB，小 %.0f 倍）" % (dst, mb, 323.0 / mb))
    print("  配套权重 %s（coef=[1.0], b=0 —— 让现有 probability() 算出 sigmoid(logit)）" % wb)


def _student_probs(pcm32k: np.ndarray, onnx_path: str, batch: int = 128):
    """学生版的滑窗概率。窗和跳与教师完全一致，
    否则「哪个候选落在哪个窗上」会错位，比的就不是同一件事了。"""
    import onnxruntime as ort
    hop = WIN_SAMPLES // 2
    x = pcm32k if len(pcm32k) >= WIN_SAMPLES else np.pad(
        pcm32k, (0, WIN_SAMPLES - len(pcm32k)))
    starts = np.arange(0, len(x) - WIN_SAMPLES + 1, hop)
    times = (starts + WIN_SAMPLES / 2.0) / PANNS_SR
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(starts), batch):
        xb = np.stack([x[s:s + WIN_SAMPLES] for s in starts[i:i + batch]]).astype(np.float32)
        # 输出名是 "embedding" 不是 "logit" —— 导出时故意沿用教师的名字，
        # 好让 Kotlin 侧那条 r[0] 的读法不用改。
        out.append(sess.run(["embedding"], {"pcm": xb})[0])
    # **必须压平。** 学生的输出是 [B,1]（故意的，为了让 Kotlin 那条读
    # Array<FloatArray> 的路径不用改），直接拿去 argsort 会沿最后一维排，
    # 一路传成二维数组，要到 hit_amplitudes 里才炸。
    lg = np.concatenate(out).ravel() if out else np.zeros(0, np.float32)
    return times, 1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30)))


def cmd_eval(args):
    """**端到端**对比，不看 AUC。

    这个项目栽过：候选级 AUC 做过四次实验，四次都和端到端结论相反
    （见 docs/RESEARCH.md）。所以这里比的是最后真正出片的那几段。
    """
    import onnxruntime as ort
    from ppai import audio as A, config, embed, highlight, rerank

    onnx_path = os.path.join(ROOT, "models", "student.onnx")
    if not os.path.exists(onnx_path):
        raise SystemExit("先跑 export")
    cfg = config.load(); hc = cfg["highlight"]
    head = rerank.load(rerank.MODEL_PATH)
    tsess = embed._load_onnx()

    paths = sources()
    for p in paths:
        pcm16 = A.extract_pcm(p, cfg["audio"]["sr"])
        hits, env, _, fr = A.detect_hits(pcm16, cfg["audio"])
        if len(hits) == 0:
            print("  %-28s 没有候选，跳过" % os.path.basename(p)); continue
        pcm32 = A.extract_pcm(p, PANNS_SR)
        dur = len(pcm32) / PANNS_SR

        tt, tf = embed.embed_windows(pcm32)
        tp = head["clf"].predict_proba(tf)[:, 1]
        st, sp = _student_probs(pcm32, onnx_path)

        # 1) 门禁：全片平均概率。阴性 0.009-0.010 / 真实 0.379-0.672，阈值 0.10
        # 2) 重排：留下来的候选是不是同一批
        # 3) 出片：最后剪出来的时间轴重合多少
        def keep(times, probs):
            idx = np.array([np.argmin(np.abs(times - c)) for c in hits])
            pr = probs[idx]
            k = max(1, int(round(len(hits) * 0.6)))
            return np.sort(hits[np.argsort(-pr, kind="stable")[:k]])

        kt, ks = keep(tt, tp), keep(st, sp)
        inter = len(np.intersect1d(kt, ks))

        def clips(kept):
            amps = A.hit_amplitudes(kept, env, fr)
            rs = highlight.score(highlight.rallies_gated(kept, dur, hc, amps),
                                 np.zeros(0), np.zeros(0), hc)
            for r in rs:
                r["src"] = p
            return highlight.select(highlight.rank(rs, "best", hc), hc,
                                    durations={p: dur})

        ct, cs = clips(kt), clips(ks)

        def union(cl):
            m = np.zeros(int(dur * 100) + 1, bool)
            for c in cl:
                m[int(c["start"] * 100):int(c["end"] * 100)] = True
            return m
        ut, us = union(ct), union(cs)
        iou = (ut & us).sum() / max((ut | us).sum(), 1)

        print("  %-28s 门禁 教师%.3f 学生%.3f | 重排重合 %d/%d | 出片 %d vs %d 段 IoU %.2f"
              % (os.path.basename(p)[:28], tp.mean(), sp.mean(),
                 inter, len(kt), len(ct), len(cs), iou), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--n", type=int, default=120000)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_data)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--width", type=float, default=1.0)
    t.add_argument("--out", default="student.pt")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("export")
    e.add_argument("--width", type=float, default=1.0)
    e.add_argument("--src", default="student.pt")
    e.set_defaults(fn=cmd_export)
    v = sub.add_parser("eval")
    v.set_defaults(fn=cmd_eval)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

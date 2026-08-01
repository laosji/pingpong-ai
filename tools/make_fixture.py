"""生成测试用的合成乒乓球素材（音频 + 视频）。

为什么不用真实录像
------------------
真实录像里是**具体的、可辨认的人**在打球，把它放进公开仓库等于替他们
做了一个不该由我们做的决定，而且开源之后会被 fork、被爬虫存档，撤不回来。

**但合成信号必须真的能触发同样的代码路径**，否则测试就是摆设。
所以这里不是随便造点噪声，而是照着这个项目实测出来的声学结构造：

  * 击球是 5-15 毫秒的宽带瞬态，能量集中在 1-5kHz
  * 回合内相邻两拍之间隔**一次落台**（用户在实拍上确认过的规律）
  * 相邻两拍间隔中位 ~0.45 秒
  * 回合之间有 1-3 秒的间隙
  * 球落地后是**连续弹跳**：间隔按固定比值收缩（恢复系数是常数），
    这是 find_bounce_decay 的判据，不造出来那段代码就测不到
  * 底噪 + 一点混响，否则自适应阈值那段（局部中位数 + k·MAD）
    在纯净信号上退化成常数阈值，测不出真实行为

生成后 tools/regen_fixtures.py 会拿它重算所有基准。
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np

SR = 32000          # 生成用 32k，需要 16k 时再降采样（PANNs 要 32k）


def transient(rng: np.random.Generator, sr: int, ms: float, lo: float, hi: float,
              amp: float) -> np.ndarray:
    """一个宽带瞬态：带通噪声 + 指数衰减包络。像一次拍击或一次触台。"""
    n = max(4, int(sr * ms / 1000.0))
    x = rng.standard_normal(n).astype(np.float32)
    # 用 FFT 做带通，避免引 scipy
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    X[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(X, n).astype(np.float32)
    env = np.exp(-np.linspace(0, 6, n)).astype(np.float32)
    x *= env
    m = np.abs(x).max()
    return (x / m * amp).astype(np.float32) if m > 0 else x


def put(buf: np.ndarray, at: float, sig: np.ndarray, sr: int) -> None:
    i = int(at * sr)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[:j - i]


def build(seconds: float = 20.0, sr: int = SR, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    # 底噪：自适应阈值靠局部中位数 + MAD，纯净信号上它会退化成常数阈值，
    # 那样就测不出真实行为了
    buf = (rng.standard_normal(n) * 0.004).astype(np.float32)

    t = 0.6
    while t < seconds - 1.0:
        # 一个回合：3-9 拍，每拍后面跟一次落台
        rally = int(rng.integers(3, 10))
        for k in range(rally):
            if t > seconds - 0.5:
                break
            # 挥拍：更响、更宽带
            put(buf, t, transient(rng, sr, rng.uniform(6, 13), 900, 6500,
                                  rng.uniform(0.45, 0.95)), sr)
            # 落台：约 0.15 秒后，弱一些、频带更窄
            bt = t + rng.uniform(0.11, 0.20)
            put(buf, bt, transient(rng, sr, rng.uniform(4, 9), 1200, 5200,
                                   rng.uniform(0.18, 0.42)), sr)
            t += rng.uniform(0.34, 0.58)      # 实测相邻两拍中位 ~0.45 秒

        # 回合结束：球落地连续弹跳。间隔按固定比值收缩 ——
        # find_bounce_decay 的判据就是「比值一致」，不造这段那里测不到。
        if rng.random() < 0.75:
            gap = rng.uniform(0.30, 0.45)
            ratio = rng.uniform(0.55, 0.72)   # 恢复系数，落在 bounce_ratio_lo/hi 之间
            bt = t
            for _ in range(6):
                put(buf, bt, transient(rng, sr, 5, 1500, 5000,
                                       rng.uniform(0.10, 0.22)), sr)
                bt += gap
                gap *= ratio
                if gap < 0.03:
                    break
            t = bt
        t += rng.uniform(1.0, 3.0)            # 回合间隙：捡球、准备

    m = np.abs(buf).max()
    return (buf / m * 0.8).astype(np.float32) if m > 0 else buf


def write_video(pcm: np.ndarray, sr: int, dst: str, w: int, h: int,
                seconds: float) -> None:
    """把音频配上一段合成画面。画面内容无所谓 —— 检测全靠声音，
    视频只是为了让切片测试有真实的容器和关键帧结构。"""
    wav = dst + ".wav"
    import wave
    with wave.open(wav, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=%dx%d:rate=30:duration=%.2f" % (w, h, seconds),
        "-i", wav,
        # 关键帧每秒一个 —— 和实测手机录像一致（6/8 段是 1.00 秒），
        # 切片测试要靠这个才有意义
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-g", "30",
        "-c:a", "aac", "-b:a", "48k", "-shortest", dst,
    ], check=True)
    os.remove(wav)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fixtures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, secs, w, h, seed in (("land", 14.0, 320, 180, 7),
                                   ("port", 10.0, 136, 240, 11)):
        pcm = build(secs, SR, seed)
        dst = os.path.join(args.out, name + ".mp4")
        write_video(pcm, SR, dst, w, h, secs)
        print("  %-10s %.0f 秒  %dx%d  %.0f KB"
              % (name + ".mp4", secs, w, h, os.path.getsize(dst) / 1024))


if __name__ == "__main__":
    main()

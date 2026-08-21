"""用合成素材重算 Kotlin 侧的全部逐点比对基准。

素材换了、模型重训了、算法改了 —— 任何一样变化都要重跑这个，
否则 Kotlin 的测试会拿旧基准比新代码，红得莫名其妙（今天就撞过两次）。

    .venv/bin/python tools/regen_fixtures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppai import audio, config, embed, highlight, rerank        # noqa: E402
from ppai.media import probe                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "android/core/src/test/resources")
ASSET = os.path.join(ROOT, "android/cutter/src/androidTest/assets")
CLIP = os.path.join(ASSET, "land.mp4")


def _select_rows(rs, h, cfg, dur):
    """select 的基准。给回合打一个假 src —— 夹片长是按来源查表的，
    而这一层的基准只有一个源。"""
    ranked = [dict(r, src="fixture") for r in highlight.rank(rs, "best", cfg)]
    return [{"start": s["start"], "end": s["end"], "hits": s["hit_count"]}
            for s in highlight.select(ranked, h, durations={"fixture": dur})]


def selfcheck() -> None:
    """跑基准之前先自检几条不变量。

    放在这里而不是单独的测试套件：Python 侧没有 pytest，而**会动音频代码的人
    正好就是要重新生成基准的人**，这一步他一定会跑到。

    1) 包络的分块必须和一次算完完全相同。分块只是为了内存
       （block=4000 帧 = 40 秒），一小时素材才不会一次吃掉几百 MB。
       而基准素材只有 14 秒（1399 帧），**跨块那条路径基准根本测不到** ——
       所以这里自己造一段足够长的信号来测。
    2) 自适应阈值的最后一块只能用真实数据算。原来用 env[-1] 重复补齐，
       那串常数把 MAD 压掉一半，片尾 2 秒的检测灵敏度凭空翻倍。
    """
    from ppai import audio as A
    cfg = config.load()["audio"]
    rng = np.random.default_rng(3)
    # 60 秒噪声加一些瞬态，保证跨 4000 帧的块边界
    n = cfg["sr"] * 60
    x = (rng.standard_normal(n) * 0.01).astype(np.float32)
    for at in rng.integers(0, n - 500, 300):
        x[at:at + 200] += rng.standard_normal(200).astype(np.float32) * 0.4

    env_blocked, _ = A.onset_envelope(x, cfg)
    src = open(os.path.join(ROOT, "ppai", "audio.py")).read()
    ns = {}
    exec(compile(src.replace("block = 4000", "block = 10**9"), "oneshot", "exec"), ns)
    env_one, _ = ns["onset_envelope"](x, cfg)
    nb = -(-len(env_blocked) // 4000)
    assert nb > 1, "自检信号没跨块，测不到要测的东西"
    d = float(np.abs(env_blocked - env_one).max())
    assert d == 0.0, "分块和一次算完不一致，最大差 %.3e" % d
    print("  自检：包络跨 %d 块，与一次算完逐帧相同" % nb)

    # 阈值不能因为末块补齐而塌掉：把信号截成半块长，看尾部阈值是否合理
    env, fr = A.onset_envelope(x, cfg)
    w = max(1, int(round(cfg["noise_win_s"] * fr)))
    cut = len(env) - w // 2                      # 故意留半块
    thr = A._adaptive_threshold(env[:cut], fr, cfg["noise_win_s"], cfg["k_mad"])
    tail, body = thr[cut - w // 2:].mean(), thr[:cut - w].mean()
    assert tail > body * 0.7, (
        "片尾阈值塌了（尾 %.2f vs 正文 %.2f）——末块八成又在拿补齐的常数算统计量"
        % (tail, body))
    print("  自检：末块只有半块时，片尾阈值 %.1f / 正文 %.1f，没有塌" % (tail, body))


def main() -> None:
    selfcheck()
    cfg = config.load()
    a, h = cfg["audio"], cfg["highlight"]
    m = rerank.load()
    keep = cfg.get("rerank", {}).get("keep_ratio", 0.6)
    os.makedirs(CORE, exist_ok=True)

    # 1) 检测层：包络 / 阈值 / 击球 / 力量
    pcm16 = audio.extract_pcm(CLIP, a["sr"])
    det, env, thr, fr = audio.detect_hits(pcm16, a)
    pcm16.astype("<f4").tofile(os.path.join(CORE, "clip.f32"))
    np.asarray(env, "<f4").tofile(os.path.join(CORE, "env.f32"))
    np.asarray(thr, "<f4").tofile(os.path.join(CORE, "thr.f32"))
    np.asarray(det, "<f8").tofile(os.path.join(CORE, "hits.f64"))
    with open(os.path.join(CORE, "params.properties"), "w") as f:
        for k, v in (("sr", a["sr"]), ("nFft", a["n_fft"]), ("hop", a["hop"]),
                     ("fmin", a["fmin"]), ("fmax", a["fmax"]), ("kMad", a["k_mad"]),
                     ("noiseWinS", a["noise_win_s"]), ("minGapS", a["min_gap_s"]),
                     ("frameRate", fr), ("nSamples", len(pcm16)),
                     ("nFrames", len(env)), ("nHits", len(det))):
            f.write("%s=%s\n" % (k, v))
    print("  检测：%d 采样 / %d 帧 / %d 次击球" % (len(pcm16), len(env), len(det)))

    # 2) 重排层
    pcm32 = audio.extract_pcm(CLIP, 32000)
    t, ft = embed.embed_windows(pcm32)
    idx = np.array([np.argmin(np.abs(t - c)) for c in det])
    p = m["clf"].predict_proba(ft[idx])[:, 1]
    k = max(1, int(round(len(det) * keep)))
    kept = np.sort(det[np.argsort(-p, kind="stable")[:k]])
    np.ascontiguousarray(pcm32).astype("<f4").tofile(os.path.join(CORE, "rr_pcm32k.f32"))
    np.asarray(det, "<f8").tofile(os.path.join(CORE, "rr_hits.f64"))
    np.asarray(p, "<f8").tofile(os.path.join(CORE, "rr_probs.f64"))
    np.asarray(kept, "<f8").tofile(os.path.join(CORE, "rr_kept.f64"))
    np.ascontiguousarray(ft[0]).astype("<f4").tofile(os.path.join(CORE, "rr_feat0.f32"))
    b = float(np.asarray(m["clf"].intercept_).ravel()[0])
    with open(os.path.join(CORE, "rr.properties"), "w") as f:
        for kk, vv in (("nWindows", len(t)), ("nHits", len(det)), ("nKept", len(kept)),
                       ("keepRatio", keep),
                       ("meanProb", "%.10f" % float(m["clf"].predict_proba(ft)[:, 1].mean())),
                       ("intercept", "%.8f" % b)):
            f.write("%s=%s\n" % (kk, vv))
    print("  重排：%d 窗，%d 候选 → 保留 %d" % (len(t), len(det), len(kept)))

    # 3) 分组层。
    #
    # **喂原始候选，不喂重排后的。** 重排器是在真实击球上训的，合成音在它
    # 看来不像击球，砍掉 40% 之后落地弹跳序列被打散，find_bounce_decay
    # 的正例分支就一次都测不到了（实测：原始候选 2 个弹跳收尾 → 重排后 0 个）。
    # 这一层要验的是**分组逻辑本身**，重排有它自己的基准（第 2 步）。
    dur = probe(CLIP)["duration"]
    d2 = det
    amps = audio.hit_amplitudes(d2, env, fr)
    rs = highlight.rallies_gated(d2, dur, h, amps)
    kind = highlight.video_kind(rs, dur, h)
    rs = highlight.score(rs, np.array([]), np.array([]), h)
    np.asarray(d2, "<f8").tofile(os.path.join(CORE, "rally_hits.f64"))
    np.asarray(amps, "<f8").tofile(os.path.join(CORE, "rally_amps.f64"))
    json.dump({
        "duration": dur, "kind": kind["kind"], "busy": kind["busy"],
        "median_gap": kind["median_gap"], "n": len(rs),
        "rallies": [{"start": round(r["start"], 6), "end": round(r["end"], 6),
                     "hits": r["hits"], "power": r["power"],
                     "peak_power": round(r["peak_power"], 6),
                     "tail_power": round(r["tail_power"], 6),
                     "ended_with_bounce": bool(r["ended_with_bounce"]),
                     "score": r["score"]} for r in rs],
        "rank_longest": [round(r["start"], 6) for r in highlight.rank(rs, "longest", cfg)[:10]],
        "rank_power": [round(r["start"], 6) for r in highlight.rank(rs, "power", cfg)[:10]],
        "rank_best": [round(r["start"], 6) for r in highlight.rank(rs, "best", cfg)[:10]],
        # 出片阶段（滤短 + 扩边 + 合并 + 夹到片长）。安卓一度整个没有这一步，
        # 直接拿 rallies 去切 —— 切点落在击球声那一帧，看不到引拍。
        # 有了这条基准，两边再走散就会当场失败。
        "select_best": _select_rows(rs, h, cfg, dur),
    }, open(os.path.join(CORE, "rally_expected.json"), "w"),
        ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(CORE, "highlight.properties"), "w") as f:
        w = h["weights"]
        for kk, vv in (("gapS", h["gap_s"]), ("minDurationS", h["min_duration_s"]),
                       ("minHits", h["min_hits"]), ("longestMinS", h["longest_min_s"]),
                       ("trimBounce", str(h["trim_bounce"]).lower()),
                       ("bounceMinCount", h["bounce_min_count"]),
                       ("bounceFinalIoi", h["bounce_final_ioi"]),
                       ("bounceRatioStd", h["bounce_ratio_std"]),
                       ("bounceRatioLo", h["bounce_ratio_lo"]),
                       ("bounceRatioHi", h["bounce_ratio_hi"]),
                       ("bounceLookaheadS", h["bounce_lookahead_s"]),
                       ("sparseGapS", h["sparse_gap_s"]), ("duration", dur),
                       ("wRallyLength", w["rally_length"]), ("wPower", w.get("power", 0.0)),
                       ("wHitRate", w["hit_rate"]), ("wMotion", w["motion"]),
                       ("wDuration", w["duration"]),
                       ("clipMinS", h["clip_min_s"]), ("padStartS", h["pad_start_s"]),
                       ("padEndS", h["pad_end_s"]),
                       ("finalPadEndS", h.get("final_pad_end_s", 0.0))):
            f.write("%s=%s\n" % (kk, vv))
    nb = sum(1 for r in rs if r["ended_with_bounce"])
    print("  分组：%d 个回合（%s），其中 %d 个以弹跳收尾" % (len(rs), kind["kind"], nb))
    # 复制权重给 Android 侧的测试用
    src = os.path.join(ROOT, "models", "rerank.bin")
    if os.path.exists(src):
        import shutil
        shutil.copy(src, os.path.join(ASSET, "rerank.bin"))


if __name__ == "__main__":
    main()

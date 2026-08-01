from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict

import numpy as np

from . import (annotate, audio, ball, config, detect, evaluate, highlight, labels,
               motion, render, rerank, scene, stats, validate, viz)
from .media import probe          # 搬到 media.py 了，这里重新导出保持兼容


def _hits(path: str, cfg: Dict):
    """抽音轨 -> 检测候选 -> （可选）重排。返回 (hits, amps)。"""
    pcm = audio.extract_pcm(path, cfg["audio"]["sr"])
    hits, env, _, fr = audio.detect_hits(pcm, cfg["audio"])
    rc = cfg.get("rerank", {})
    if rc.get("enabled") and len(hits):
        model = rerank.load(rc.get("model", rerank.MODEL_PATH))
        if model is not None:
            before = len(hits)
            hits = rerank.apply(path, hits, model, rc.get("keep_ratio", 0.6))
            print("  重排: %d -> %d 个候选（模型训练自 %d 份标注）"
                  % (before, len(hits), len(model["files"])))
    return hits, audio.hit_amplitudes(hits, env, fr)


def analyze(path: str, cfg: Dict, out_dir: str, make_plot: bool = True) -> Dict:
    meta = probe(path)
    t0 = time.time()

    pcm = audio.extract_pcm(path, cfg["audio"]["sr"])
    audio_dur = len(pcm) / float(cfg["audio"]["sr"])
    hits, env, thr, frame_rate = audio.detect_hits(pcm, cfg["audio"])
    print("  音频: %d 个声学瞬态 (%.1fs 音轨)" % (len(hits), audio_dur))
    # 抽出来的音轨远短于视频 = 文件损坏或抽取失败，后面所有结论都会是垃圾
    if meta["duration"] > 0 and audio_dur < meta["duration"] * 0.9:
        print("  ⚠️  音轨仅 %.1fs 而视频 %.1fs —— 文件可能损坏，结果不可信"
              % (audio_dur, meta["duration"]))

    m_t, m_v = motion.motion_curve(path, cfg["motion"])
    print("  运动: %d 个采样点" % len(m_t))

    fused = detect.fuse(hits, m_t, m_v, meta["duration"], cfg["fuse"])
    scene_score = scene.assess(path, cfg.get("scene", {}))
    if cfg.get("scene", {}).get("enabled", True):
        fused["score"] = fused["score"] * scene_score["score"]
        print("  场景: score %.2f, edge %.3f%s" % (
            scene_score["score"], scene_score["edge"],
            "" if scene_score["passed"] else "，疑似非乒乓球场景"))
    segments = detect.segment(fused["grid"], fused["score"], meta["duration"], hits, cfg["fuse"])

    covered = sum(s["duration"] for s in segments)
    result = {
        "video": path,
        "meta": meta,
        "params": cfg,
        "hits": [round(float(x), 3) for x in hits],
        "scene": scene_score,
        "segments": segments,
        "summary": {
            "segment_count": len(segments),
            "covered_seconds": round(covered, 2),
            "covered_ratio": round(covered / meta["duration"], 3) if meta["duration"] else 0.0,
            "analyze_seconds": round(time.time() - t0, 1),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0][:40]
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if make_plot:
        viz.plot(os.path.join(out_dir, stem + ".png"), os.path.basename(path),
                 env, thr, frame_rate, hits, fused, segments, cfg)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ppai", description="乒乓球有效比赛检测原型")
    p.add_argument("command", choices=["probe", "analyze", "cut", "validate",
                                       "annotate", "label-negative", "eval", "eval-hits",
                                       "highlight", "stats", "themes",
                                       "rerank-train"])
    p.add_argument("paths", nargs="*")
    p.add_argument("--pos", help="validate: 正样本 glob")
    p.add_argument("--neg", help="validate: 阴性对照 glob")
    p.add_argument("--labels", default="labels", help="标注目录（默认 labels/）")
    p.add_argument("--window", nargs=2, type=float, metavar=("START","END"),
                   help="annotate: 只标注这段（秒），并声明为穷尽标注区间")
    p.add_argument("-c", "--config", default=None)
    p.add_argument("-o", "--out", default="out")
    p.add_argument("-s", "--set", action="append", dest="overrides",
                   help="覆盖配置，如 -s audio.k_mad=3.0 -s fuse.enter=0.4")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-dedupe", action="store_true",
                   help="highlight: 允许不同主题之间出现重复片段")
    p.add_argument("--top", type=int, help="highlight: 取前几个回合")
    p.add_argument("--minutes", type=float, help="highlight: 目标集锦时长（分钟）")
    p.add_argument("--type", default="auto",
                   choices=["auto", "best", "longest", "power",
                            "trim", "all"],
                   help="highlight: 集锦类型（对应方案模块七的四种）")
    args = p.parse_args(argv)

    cfg = config.override(config.load(args.config), args.overrides)

    if args.command == "rerank-train":
        rerank.train(args.labels, cfg)
        return 0

    if args.command == "themes":
        # 前端拉这个列表渲染选项
        print(json.dumps(highlight.THEMES, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate":
        if not args.pos or not args.neg:
            p.error("validate 需要 --pos 和 --neg")
        res = validate.run(args.pos, args.neg, cfg)
        return 0 if res["passed"] else 1

    if args.command == "eval-hits":
        for v in args.paths:
            lab = labels.find(v, args.labels)
            if lab is None:
                print("没有标注: %s" % os.path.basename(v)); continue
            ranges = labels.scored_ranges(lab)
            if not ranges:
                print("标注没有 complete_ranges，无法算准确率（抽样标注只能算召回）")
                continue
            # 击球被拖成短区间存放，取中点；F 键标的 hits 一并计入
            truth = sorted([ (a+b)/2.0 for a, b in lab["playing"] if b-a <= 1.0 ]
                           + list(lab.get("hits", [])))
            pcm = audio.extract_pcm(v, cfg["audio"]["sr"])
            det, _, _, _ = audio.detect_hits(pcm, cfg["audio"])
            r = evaluate.hit_level(np.array(det), np.array(truth), ranges)
            span = sum(b-a for a, b in ranges)
            print("\n>> %s" % os.path.basename(v))
            print("   穷尽标注区间: %s  合计 %.0f 秒"
                  % (", ".join("%.0f-%.0f" % (a, b) for a, b in ranges), span))
            print("   人工 %d 次击球 (%.2f 次/秒) | 检出 %d 个瞬态 (%.2f 次/秒)"
                  % (r["n_truth"], r["n_truth"]/span, r["n_det"], r["n_det"]/span))
            print("   召回 %.3f   准确 %.3f   F1 %.3f" % (r["recall"], r["precision"], r["f1"]))
            if r["precision"] == r["precision"] and r["precision"] < 0.5:
                print("   -> 每 1 个真击球伴随约 %.1f 个误报"
                      % (1.0/max(r["precision"], 1e-6) - 1))
        return 0

    if args.command == "eval":
        res = evaluate.evaluate(
            args.paths, args.labels,
            lambda v: analyze(v, cfg, args.out, make_plot=False)["segments"])
        evaluate.report(res)
        return 0

    for path in args.paths:
        if not os.path.exists(path):
            print("跳过（不存在）: %s" % path, file=sys.stderr)
            continue
        print("\n>> %s" % os.path.basename(path))

        if args.command == "probe":
            print(json.dumps(probe(path), ensure_ascii=False, indent=2))
            continue

        if args.command == "highlight":
            hcfg = dict(cfg["highlight"])
            if args.top:
                hcfg["top_n"] = args.top
            meta = probe(path)
            hits, amps = _hits(path, cfg)
            m_t, m_v = motion.motion_curve(path, cfg["motion"])
            rs = highlight.score(
                highlight.rallies_gated(hits, meta["duration"], hcfg, amps),
                m_t, m_v, hcfg)
            vk = highlight.video_kind(rs, meta["duration"], hcfg)
            print("  %d 个瞬态 -> %d 个回合 | 结构: %s（空档中位 %.1fs, 忙碌 %.0f%%）"
                  % (len(hits), len(rs),
                     "稀疏，可大幅压缩" if vk["kind"] == "sparse" else "密集，删不掉多少",
                     vk["median_gap"], vk["busy"] * 100))
            if args.type == "auto":
                kinds = ["trim"]      # 自动 = 完整版，与网页端/skill 保持一致
                print("  自动 = 完整版：去掉捡球和等待，一个球都不漏")
            elif args.type == "all":
                kinds = ["best", "longest", "power"]
            else:
                kinds = [args.type]
                # 用户手选了不适合这段素材的主题：不拒绝，但要说明会得到什么
                th = next((t for t in highlight.THEMES if t["id"] == args.type), None)
                if th and th["applicable"] and vk["kind"] not in th["applicable"]:
                    print("  提示: 「%s」更适合%s素材；这段是%s，"
                          % (th["name"],
                             "稀疏（有大量捡球可删）" if "sparse" in th["applicable"] else "密集",
                             "密集（几乎全在打球）" if vk["kind"] == "dense" else "稀疏")
                          + "结果仍可用但压缩比会很低。")
            stem = os.path.splitext(os.path.basename(path))[0][:40]
            used: list = []
            for kind in kinds:
                if kind == "trim":
                    # 完整版不走排序/取前N —— 它要的是全覆盖，不是挑最好的
                    picked = highlight.trim_idle(hits, meta["duration"], hcfg)
                else:
                    picked = highlight.select(highlight.rank(rs, kind, hcfg), hcfg,
                                              args.minutes * 60 if args.minutes else None)
                if len(kinds) > 1 and not args.no_dedupe:
                    before = len(picked)
                    picked = highlight.dedupe(picked, used, hcfg)
                    if before != len(picked):
                        print("\n  [%s] 去掉 %d 个与前面主题重叠的片段"
                              % (kind, before - len(picked)))
                used.extend(picked)
                print("\n  [%s] %s" % (kind, highlight.RANKERS[kind]))
                if not picked:
                    print("    没有符合条件的片段"); continue
                for s_ in picked:
                    print("    #%-2d %6.1f-%6.1fs (%4.1fs, %3d个瞬态, 力量 %3.0f, 收尾 %3.0f, 比值 %.2f)"
                          % (s_["id"], s_["start"], s_["end"], s_["duration"],
                             s_["hit_count"], s_["power"], s_["tail_power"],
                             s_["tail_power"] / max(s_["power"], 1e-6)))
                parts = render.cut(path, picked,
                                   os.path.join(args.out, "%s_hl_%s" % (stem, kind)),
                                   cfg["render"])
                final = render.concat(
                    parts, os.path.join(args.out, "%s_%s.mp4" % (stem, kind)))
                total = sum(s_["duration"] for s_ in picked)
                print("    输出: %s  (%.0f 秒)" % (final, total))
            continue

        if args.command == "stats":
            meta = probe(path)
            hits, amps = _hits(path, cfg)
            hcfg = cfg["highlight"]
            m_t, m_v = motion.motion_curve(path, cfg["motion"])
            rs = highlight.score(
                highlight.rallies_gated(hits, meta["duration"], hcfg, amps),
                m_t, m_v, hcfg)
            vk = highlight.video_kind(rs, meta["duration"], hcfg)
            s_ = stats.summarize(rs, hits, amps, meta["duration"], vk,
                                 busy_gap=cfg["highlight"].get("busy_gap_s", 0.3))
            stats.report(s_)
            os.makedirs(args.out, exist_ok=True)
            dst = os.path.join(args.out, os.path.splitext(
                os.path.basename(path))[0][:40] + "_stats.json")
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(s_, f, ensure_ascii=False, indent=2)
            print("\n  已保存: %s" % dst)
            continue

        if args.command == "label-negative":
            meta = probe(path)
            lab = labels.negative(path, meta["duration"])
            dst = labels.save(lab, labels.path_for(path, args.labels))
            print("  已标为阴性（全程无乒乓球）: %s" % dst)
            continue

        if args.command == "annotate":
            meta = probe(path)
            existing = labels.find(path, args.labels)
            lab = existing or labels.empty(path, meta["duration"])
            if existing:
                print("  载入已有标注: %d 段" % len(lab["playing"]))
            # 叠加 AI 预测供参考，人工只需修正 —— 方案模块八的数据闭环入口
            pred = analyze(path, cfg, args.out, make_plot=False)["segments"]
            stem = os.path.splitext(os.path.basename(path))[0]
            win = tuple(args.window) if args.window else None
            if win:
                stem += "_%d-%d" % (int(win[0]), int(win[1]))
            dst = annotate.build(path, meta["duration"],
                                 os.path.join(args.labels, stem + ".html"), lab, pred,
                                 window=win)
            print("  标注页面: %s" % dst)
            print("  标完点「导出 JSON」，把文件存到 %s/" % args.labels)
            continue

        res = analyze(path, cfg, args.out, make_plot=not args.no_plot)
        s = res["summary"]
        print("  片段: %d 个, 覆盖 %.1fs / %.1fs (%.0f%%), 耗时 %.1fs"
              % (s["segment_count"], s["covered_seconds"], res["meta"]["duration"],
                 s["covered_ratio"] * 100, s["analyze_seconds"]))
        for seg in res["segments"]:
            print("    #%-2d %7.2f - %7.2f  (%5.1fs, %3d拍, conf %.2f)"
                  % (seg["id"], seg["start"], seg["end"], seg["duration"],
                     seg["hit_count"], seg["confidence"]))

        if args.command == "cut":
            if not res["segments"]:
                print("  无片段，跳过剪辑")
                continue
            stem = os.path.splitext(os.path.basename(path))[0][:40]
            seg_dir = os.path.join(args.out, stem + "_segments")
            parts = render.cut(path, res["segments"], seg_dir, cfg["render"])
            final = render.concat(parts, os.path.join(args.out, stem + "_highlight.mp4"))
            print("  输出: %s" % final)

    return 0


if __name__ == "__main__":
    sys.exit(main())

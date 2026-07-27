"""验证工具：正样本 + 阴性对照。

存在的理由：第一版检测器在 6 个乒乓球视频上给出 100% 覆盖、置信度 1.00，
看起来完美 —— 直到把一段有声小说朗读喂进去，它同样给出 100% 覆盖。
只看正样本永远发现不了这个问题。

任何新的检测方案，先过这一关再谈准确率。
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

from . import audio, detect, motion


def _measure(path: str, cfg: Dict) -> Dict:
    """跑到片段这一步。真正要看的是覆盖率 —— 检测器判定「在打球」的时间占比。

    击球率不是好指标：它只说明检测器找到了多少瞬态，不说明它敢不敢说「没在打」。
    """
    x = audio.extract_pcm(path, cfg["audio"]["sr"])
    dur = len(x) / float(cfg["audio"]["sr"])
    if dur < 1.0:
        return {"duration": dur, "hits": 0, "rate": 0.0, "coverage": 0.0, "valid": False}

    hits, _, _, _ = audio.detect_hits(x, cfg["audio"])
    m_t, m_v = motion.motion_curve(path, cfg["motion"])
    fused = detect.fuse(hits, m_t, m_v, dur, cfg["fuse"])
    segs = detect.segment(fused["grid"], fused["score"], dur, hits, cfg["fuse"])
    covered = sum(s["duration"] for s in segs)
    return {"duration": dur, "hits": len(hits), "rate": len(hits) / dur,
            "coverage": covered / dur, "valid": True}


def run(pos_glob: str, neg_glob: str, cfg: Dict) -> Dict:
    pos = sorted(glob.glob(pos_glob))
    neg = sorted(glob.glob(neg_glob))
    if not pos or not neg:
        raise SystemExit("正样本或阴性样本为空：pos=%d neg=%d" % (len(pos), len(neg)))

    rows: List[Dict] = []
    for kind, files in (("正", pos), ("阴性", neg)):
        for p in files:
            r = _measure(p, cfg)
            r.update(kind=kind, name=os.path.basename(p)[:34])
            rows.append(r)

    print("%-6s %-36s %8s %6s %8s %8s" % ("类别", "文件", "时长", "瞬态", "次/秒", "覆盖率"))
    for r in rows:
        print("%-6s %-36s %7.1fs %6d %7.2f %7.0f%%%s"
              % (r["kind"], r["name"], r["duration"], r["hits"], r["rate"],
                 r["coverage"] * 100, "" if r["valid"] else "  <- 音轨缺失，样本无效"))

    valid = [r for r in rows if r["valid"]]
    p_cov = [r["coverage"] for r in valid if r["kind"] == "正"]
    n_cov = [r["coverage"] for r in valid if r["kind"] == "阴性"]
    if not p_cov or not n_cov:
        raise SystemExit("\n有效样本不足")

    margin = min(p_cov) - max(n_cov)
    print("\n正样本最低覆盖 %.0f%% | 阴性最高覆盖 %.0f%% | 间隔 %+.0f 个百分点"
          % (min(p_cov) * 100, max(n_cov) * 100, margin * 100))
    # 阴性素材完全没有乒乓球，覆盖率理应接近 0；只要超过 20% 就是在瞎判
    worst_neg = max(n_cov)
    passed = margin > 0.25 and worst_neg < 0.20
    if passed:
        print("通过。")
    else:
        print("未通过：阴性素材（无乒乓球）被判定 %.0f%% 时间在打球。" % (worst_neg * 100))
        print("      检测器对「有结构的音频」一律触发，并没有在识别击球声。")
    return {"rows": rows, "margin": margin, "passed": passed}

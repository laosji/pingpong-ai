"""媒体探测 —— ffprobe 封装 + 素材质量结论。

**独立成模块是为了桌面 app。** 原来 probe() 住在 cli.py 里，而 cli.py 顶部
一次性 import 了 viz / annotate / ball / evaluate / validate 这些开发工具，
它们又拖着 matplotlib（33MB）、sklearn（47MB）、scipy（98MB）。
app 只需要探测和剪辑，为一个 ffprobe 封装背 180MB 依赖没有道理。

cli.py 仍然 `from .media import probe` 重新导出，老的调用点不用改。
"""
from __future__ import annotations

import json
import subprocess
from typing import Dict


def probe(path: str) -> Dict:
    """对应方案「模块二：视频质量检测」。"""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = json.loads(out.stdout or b"{}")
    v = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), None)
    a = next((s for s in info.get("streams", []) if s["codec_type"] == "audio"), None)
    if v is None:
        raise RuntimeError("没有视频流: %s" % path)

    num, _, den = v.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den or 1)
    w, h = int(v["width"]), int(v["height"])
    duration = float(info.get("format", {}).get("duration", 0.0))

    # 提示要给结论，不能只报参数。我们靠**击球声**判断回合，所以分辨率、
    # 帧率、方向都不影响剪辑准确度，只影响成片观感；只有没音轨是致命的。
    # 旧文案把这些一视同仁地扣分，一段完全能剪的素材会显示 55 分，
    # 让用户以为结果会很差。
    notes, notes_en = [], []
    score, blocking = 100, False
    if a is None:
        score -= 70
        blocking = True
        notes.append("没有音轨 —— 我们靠击球声判断回合，这段没法自动剪")
        notes_en.append("No audio track — rallies are detected from hit sounds, "
                        "so this clip can't be edited automatically")
    if min(w, h) < 720:
        score -= 8
        notes.append("分辨率 %d×%d 偏低 —— 不影响剪得准不准（靠声音判断），"
                     "只是成片清晰度一般" % (w, h))
        notes_en.append("Low resolution (%d×%d) — detection is audio-based and "
                        "unaffected; only the output looks softer" % (w, h))
    if fps < 25:
        score -= 5
        notes.append("帧率 %.0f fps 偏低 —— 同样不影响剪辑，只影响画面流畅度" % fps)
        notes_en.append("Low frame rate (%.0f fps) — also doesn't affect the edit, "
                        "only smoothness" % fps)
    if h > w:
        score -= 5
        notes.append("竖屏拍摄 —— 不影响剪辑；和横屏素材一起剪时会自动加黑边对齐")
        notes_en.append("Portrait video — fine to edit; it gets letterboxed "
                        "automatically when combined with landscape clips")
    if duration < 120:
        notes.append("不到 2 分钟，像是已经剪过的片子 —— 能剪，但可压缩的空间不多")
        notes_en.append("Under 2 minutes — looks already edited; there may not be "
                        "much idle time left to cut")

    return {
        "path": path, "width": w, "height": h, "fps": round(fps, 2),
        "duration": round(duration, 2), "has_audio": a is not None,
        "quality_score": max(0, score),
        "blocking": blocking,
        "recommendation": "；".join(notes) or "素材没问题，可以直接剪",
        "recommendation_en": "; ".join(notes_en) or "Good to go",
    }

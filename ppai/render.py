"""切片 + 拼接输出。对应方案「模块七」。"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, List, Optional


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-2000:])


_DIMS: Dict[str, tuple] = {}

# 叠字用的字体。必须能显示中文 —— 英文字体会把汉字画成方框。
# 按平台列候选，一个都找不到就**跳过叠字**而不是报错：
# 部署环境（Docker）里往往没有系统中文字体，不该让整个出片失败。
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Songti.ttc",       # macOS
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Debian/Ubuntu
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",        # Alpine/Arch
)
_FONT: Optional[str] = None
_FONT_LOOKED = False


def caption_font() -> Optional[str]:
    global _FONT, _FONT_LOOKED
    if not _FONT_LOOKED:
        _FONT_LOOKED = True
        _FONT = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
    return _FONT


def _esc(text: str) -> str:
    """drawtext 的文本要转义 —— 冒号和反斜杠是它的语法字符。"""
    return (text.replace("\\", r"\\\\").replace(":", r"\\:")
                .replace("'", "").replace("%", r"\\%"))


def _caption(text: str, h: int) -> str:
    """左下角的一行说明，带半透明底 —— 直接白字在浅色地板上会看不见。"""
    fp = caption_font()
    if not fp or not text:
        return ""
    size = max(14, int(h * 0.042))
    pad = max(6, size // 3)
    return ("drawtext=fontfile=%s:text='%s':fontsize=%d:fontcolor=white"
            ":box=1:boxcolor=black@0.45:boxborderw=%d:x=%d:y=h-th-%d"
            % (fp, _esc(text), size, pad, pad * 2, pad * 2))


def dims(path: str) -> tuple:
    """视频的宽高，带缓存 —— 多选时同一个源会被反复问到。"""
    if path not in _DIMS:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            w, h = p.stdout.decode().strip().split("x")[:2]
            _DIMS[path] = (int(w), int(h))
        except Exception:
            _DIMS[path] = (0, 0)
    return _DIMS[path]


def pick_canvas(paths: List[str], weights: Dict[str, float] = None) -> tuple:
    """多个源混剪时的统一画布。

    concat 用 -c copy，要求所有片段编码参数一致。横竖混选时不统一画布，
    拼出来的是**可变分辨率流** —— ffmpeg 不报错，但容器只声明第一段的尺寸，
    播放器会把后面方向不同的段排版错（实测 960x544 + 544x960 拼完，
    容器写 960x544，竖屏那半段被按横屏排）。

    按总时长选多数派方向，让大部分内容保持原生尺寸、少数派加黑边。
    """
    if not paths:
        return (0, 0)
    tally: Dict[tuple, float] = {}
    for p in paths:
        d = dims(p)
        if d[0]:
            tally[d] = tally.get(d, 0.0) + (weights or {}).get(p, 1.0)
    return max(tally.items(), key=lambda kv: kv[1])[0] if tally else dims(paths[0])


def _fit(src_dims: tuple, canvas: tuple) -> str:
    """把源缩放进画布并居中补黑边。尺寸本来就一致时返回空串，不加多余滤镜。"""
    if not canvas or not canvas[0] or src_dims == canvas:
        return ""
    w, h = canvas
    return ("scale=%d:%d:force_original_aspect_ratio=decrease,"
            "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black,setsar=1" % (w, h, w, h))


def cut(video_path: str, segments: List[Dict], out_dir: str, cfg: Dict,
        fade_last: bool = True, canvas: tuple = None, on_progress=None) -> List[str]:
    """切片。最后一段做淡出。

    为什么是淡出而不是调尾巴长度：结尾「仓促」的根源是**硬切** ——
    画面和声音在某一帧突然消失。不管留 0.5 秒还是 1.5 秒都一样刺耳，
    调常数只是在两种难受之间挪。淡出给观众一个「要结束了」的信号，
    问题就消失了，而且尾巴留多久变得不敏感。

    音频淡出比画面更重要：突然静音比突然黑屏刺耳得多。

    多素材：每段可以带自己的 `src`（不带就用 video_path），配合 canvas
    统一到同一尺寸。反正每段本来就要重编码，加缩放滤镜不多一次转码。
    """
    os.makedirs(out_dir, exist_ok=True)
    # 清掉上次运行的残留：这次片段变少时旧文件会留在目录里，
    # 谁要是直接拼接整个目录就会拿到多余的片段。
    for f in os.listdir(out_dir):
        if f.startswith("seg_") and f.endswith(".mp4"):
            os.unlink(os.path.join(out_dir, f))
    fade = float(cfg.get("fade_out_s", 0.0))
    sm = cfg.get("slowmo") or {}
    paths = []
    for i, s in enumerate(segments):
        dst = os.path.join(out_dir, "seg_%03d.mp4" % s["id"])
        src = s.get("src") or video_path
        fit = _fit(dims(src), canvas)
        cap = _caption(s.get("label", ""), (canvas or dims(src))[1] or 720)
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-ss", str(s["start"]), "-i", src, "-t", str(s["duration"]),
               "-c:v", "libx264", "-preset", cfg["preset"], "-crf", str(cfg["crf"]),
               "-c:a", "aac", "-b:a", cfg["audio_bitrate"]]
        # 慢动作：只放慢最后一拍前后的一小段，不动整个片段。
        # 位置来自 last_hit（击球时间戳），不是片段中点 —— 猜不准就没有意义。
        if sm.get("enabled") and s.get("last_hit"):
            rel = s["last_hit"] - s["start"]
            a = max(0.0, rel - float(sm.get("lead_s", 0.35)))
            b = min(s["duration"], rel + float(sm.get("tail_s", 0.55)))
            f = float(sm.get("factor", 0.6))
            if b - a > 0.15:
                vf = ("[0:v]trim=0:%.3f,setpts=PTS-STARTPTS[a];"
                      "[0:v]trim=%.3f:%.3f,setpts=(PTS-STARTPTS)/%.4f%s[b];"
                      "[0:v]trim=%.3f,setpts=PTS-STARTPTS[c];"
                      "[a][b][c]concat=n=3:v=1[v]") % (
                    a, a, b, f,
                    ",minterpolate=fps=60:mi_mode=mci" if sm.get("smooth") else "",
                    b)
                af = ("[0:a]atrim=0:%.3f,asetpts=PTS-STARTPTS[x];"
                      "[0:a]atrim=%.3f:%.3f,asetpts=PTS-STARTPTS,atempo=%.4f[y];"
                      "[0:a]atrim=%.3f,asetpts=PTS-STARTPTS[z];"
                      "[x][y][z]concat=n=3:v=0:a=1[aud]") % (a, a, b, f, b)
                extra = ",".join(x for x in (fit, cap) if x)
                if extra:    # 要接在慢动作链末尾，不能另开 -vf
                    vf = vf.replace("concat=n=3:v=1[v]",
                                    "concat=n=3:v=1," + extra + "[v]")
                cmd = cmd[:cmd.index("-c:v")] + [
                    "-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[aud]",
                ] + cmd[cmd.index("-c:v"):]
                fit = cap = ""   # 已并进 filter_complex，别再挂一次 -vf
        # -vf 只能出现一次，统一画布和淡出必须串成一条链
        # 顺序有讲究：先统一画布再叠字，否则字会跟着缩放一起被拉伸
        chain = [x for x in (fit, cap) if x]
        last = fade_last and fade > 0 and i == len(segments) - 1
        if last:
            st = max(0.0, s["duration"] - fade)
            chain.append("fade=t=out:st=%.3f:d=%.3f" % (st, fade))
            cmd += ["-af", "afade=t=out:st=%.3f:d=%.3f" % (st, fade)]
        if chain:
            cmd += ["-vf", ",".join(chain)]
        cmd += ["-avoid_negative_ts", "make_zero", dst]
        _run(cmd)
        paths.append(dst)
        if on_progress:
            on_progress(i + 1, len(segments))
    return paths


def concat(parts: List[str], out_path: str) -> str:
    if not parts:
        raise ValueError("没有可拼接的片段")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % os.path.abspath(p).replace("'", r"'\''"))
        listfile = f.name
    try:
        _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
              "-i", listfile, "-c", "copy", out_path])
    finally:
        os.unlink(listfile)
    return out_path

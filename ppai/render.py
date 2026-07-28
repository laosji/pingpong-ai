"""切片 + 拼接输出。对应方案「模块七」。"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, List


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-2000:])


def cut(video_path: str, segments: List[Dict], out_dir: str, cfg: Dict,
        fade_last: bool = True) -> List[str]:
    """切片。最后一段做淡出。

    为什么是淡出而不是调尾巴长度：结尾「仓促」的根源是**硬切** ——
    画面和声音在某一帧突然消失。不管留 0.5 秒还是 1.5 秒都一样刺耳，
    调常数只是在两种难受之间挪。淡出给观众一个「要结束了」的信号，
    问题就消失了，而且尾巴留多久变得不敏感。

    音频淡出比画面更重要：突然静音比突然黑屏刺耳得多。
    """
    os.makedirs(out_dir, exist_ok=True)
    # 清掉上次运行的残留：这次片段变少时旧文件会留在目录里，
    # 谁要是直接拼接整个目录就会拿到多余的片段。
    for f in os.listdir(out_dir):
        if f.startswith("seg_") and f.endswith(".mp4"):
            os.unlink(os.path.join(out_dir, f))
    fade = float(cfg.get("fade_out_s", 0.0))
    paths = []
    for i, s in enumerate(segments):
        dst = os.path.join(out_dir, "seg_%03d.mp4" % s["id"])
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-ss", str(s["start"]), "-i", video_path, "-t", str(s["duration"]),
               "-c:v", "libx264", "-preset", cfg["preset"], "-crf", str(cfg["crf"]),
               "-c:a", "aac", "-b:a", cfg["audio_bitrate"]]
        if fade_last and fade > 0 and i == len(segments) - 1:
            st = max(0.0, s["duration"] - fade)
            cmd += ["-vf", "fade=t=out:st=%.3f:d=%.3f" % (st, fade),
                    "-af", "afade=t=out:st=%.3f:d=%.3f" % (st, fade)]
        cmd += ["-avoid_negative_ts", "make_zero", dst]
        _run(cmd)
        paths.append(dst)
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

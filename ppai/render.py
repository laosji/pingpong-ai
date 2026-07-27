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


def cut(video_path: str, segments: List[Dict], out_dir: str, cfg: Dict) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    # 清掉上次运行的残留：这次片段变少时旧文件会留在目录里，
    # 谁要是直接拼接整个目录就会拿到多余的片段。
    for f in os.listdir(out_dir):
        if f.startswith("seg_") and f.endswith(".mp4"):
            os.unlink(os.path.join(out_dir, f))
    paths = []
    for s in segments:
        dst = os.path.join(out_dir, "seg_%03d.mp4" % s["id"])
        _run([
            "ffmpeg", "-v", "error", "-y",
            "-ss", str(s["start"]), "-i", video_path, "-t", str(s["duration"]),
            "-c:v", "libx264", "-preset", cfg["preset"], "-crf", str(cfg["crf"]),
            "-c:a", "aac", "-b:a", cfg["audio_bitrate"],
            "-avoid_negative_ts", "make_zero", dst,
        ])
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

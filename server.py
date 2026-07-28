"""剪辑页面的后端。直接调现有管线，不复制逻辑。

任务放线程池异步跑：一个 12 分钟的视频要 10-30 秒，同步接口会把浏览器挂住。
进度只做粗粒度（分析/剪辑/完成）—— 管线内部没有进度回调，
硬编码百分比会是假的，不如只报当前阶段。
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
import traceback
import uuid
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ppai import audio, config, highlight, motion, render, rerank, stats
from ppai.cli import probe

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
UPLOADS = os.path.join(ROOT, "uploads")
VIDEO_DIRS = [os.path.expanduser("~/Downloads/PP-video"), UPLOADS]
ALLOWED_EXT = (".mp4", ".mov", ".m4v")

app = FastAPI(title="乒乓球集锦")
_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()


class Job(BaseModel):
    video: str
    themes: List[str] = ["auto"]
    top: int = 10


def _fingerprint(path: str) -> str:
    """文件指纹：大小 + 首尾各 1MB 的哈希。整文件哈希对几十上百 MB 的视频太慢，
    而同一个视频的首尾字节几乎不可能碰撞。"""
    size = os.path.getsize(path)
    h = hashlib.blake2b(str(size).encode(), digest_size=16)
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()


def _list_videos() -> List[Dict]:
    out, seen = [], set()
    for d in VIDEO_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(ALLOWED_EXT):
                continue
            p = os.path.join(d, f)
            try:
                fp = _fingerprint(p)
                if fp in seen:      # 同一个视频既在素材目录又在 uploads，只列一次
                    continue
                m = probe(p)
            except Exception:
                continue
            seen.add(fp)
            out.append({"path": p, "name": f, "duration": m["duration"],
                        "width": m["width"], "height": m["height"],
                        "quality": m["quality_score"], "note": m["recommendation"],
                        "fp": fp})
    return out


def _run(job_id: str, req: Job) -> None:
    def step(msg: str) -> None:
        with _lock:
            _jobs[job_id]["stage"] = msg
    try:
        cfg = config.load()
        hcfg = dict(cfg["highlight"])
        hcfg["top_n"] = req.top
        step("读取音轨")
        pcm = audio.extract_pcm(req.video, cfg["audio"]["sr"])
        hits, env, _, fr = audio.detect_hits(pcm, cfg["audio"])
        rc = cfg.get("rerank", {})
        if rc.get("enabled") and len(hits):
            model = rerank.load(rc.get("model", rerank.MODEL_PATH))
            if model is not None:
                step("重排候选")
                hits = rerank.apply(req.video, hits, model, rc.get("keep_ratio", 0.6))
        amps = audio.hit_amplitudes(hits, env, fr)
        step("分析画面运动")
        m_t, m_v = motion.motion_curve(req.video, cfg["motion"])
        rs = highlight.score(highlight.rallies(hits, hcfg, amps), m_t, m_v, hcfg)
        meta = probe(req.video)
        vk = highlight.video_kind(rs, meta["duration"], hcfg)
        summary = stats.summarize(rs, hits, amps, meta["duration"], vk,
                                  busy_gap=hcfg.get("busy_gap_s", 0.3))

        kinds = req.themes
        if kinds == ["auto"]:
            # 页面是单选，自动模式也只出一个 —— 取该结构下推荐的第一个主题
            kinds = highlight.AUTO_THEMES.get(vk["kind"], ["best"])[:1]
        stem = os.path.splitext(os.path.basename(req.video))[0][:40]
        results, used = [], []
        for kind in kinds:
            step("剪辑：%s" % highlight.RANKERS.get(kind, kind).split(" ")[0])
            picked = highlight.select(highlight.rank(rs, kind, hcfg), hcfg)
            if len(kinds) > 1:
                picked = highlight.dedupe(picked, used, hcfg)
            used.extend(picked)
            if not picked:
                results.append({"theme": kind, "empty": True})
                continue
            seg_dir = "%s_hl_%s" % (stem, kind)
            parts = render.cut(req.video, picked, os.path.join(OUT, seg_dir), cfg["render"])
            final = render.concat(parts, os.path.join(OUT, "%s_%s.mp4" % (stem, kind)))
            # 每个片段单独可下载/可跳转。offset 是它在拼接成片里的起点，
            # 前端据此把主播放器 seek 过去。
            off = 0.0
            for s_, part in zip(picked, parts):
                s_["url"] = "/out/%s/%s" % (seg_dir, os.path.basename(part))
                s_["offset"] = round(off, 2)
                off += s_["duration"]
            results.append({
                "theme": kind,
                "name": next((t["name"] for t in highlight.THEMES if t["id"] == kind), kind),
                "url": "/out/" + os.path.basename(final),
                "seconds": round(sum(s["duration"] for s in picked), 1),
                "clips": len(picked),
                "segments": picked,
            })
        with _lock:
            _jobs[job_id].update(state="done", stage="完成", results=results,
                                 stats=summary, kind=vk)
    except Exception as e:
        traceback.print_exc()
        with _lock:
            _jobs[job_id].update(state="error", stage="失败", error=str(e))


@app.get("/api/videos")
def videos():
    return _list_videos()


@app.get("/api/themes")
def themes():
    return highlight.THEMES


@app.post("/api/jobs")
def create(req: Job):
    if not os.path.exists(req.video):
        raise HTTPException(404, "视频不存在")
    jid = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[jid] = {"id": jid, "state": "running", "stage": "排队中",
                      "video": req.video, "started": time.time()}
    threading.Thread(target=_run, args=(jid, req), daemon=True).start()
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    with _lock:
        j = _jobs.get(jid)
    if not j:
        raise HTTPException(404, "任务不存在")
    return JSONResponse(j)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "只支持 %s" % "/".join(ALLOWED_EXT))
    # 只取文件名，丢掉任何路径成分 —— 上传的 filename 是客户端给的，不可信
    safe = os.path.basename(file.filename or "video" + ext).replace("/", "_")
    os.makedirs(UPLOADS, exist_ok=True)
    dst = os.path.join(UPLOADS, safe)
    n = 1
    while os.path.exists(dst):
        stem, e = os.path.splitext(safe)
        dst = os.path.join(UPLOADS, "%s_%d%s" % (stem, n, e)); n += 1
    with open(dst, "wb") as fh:
        while chunk := await file.read(1 << 20):     # 分块写，避免整个视频进内存
            fh.write(chunk)
    try:
        m = probe(dst)
    except Exception:
        os.unlink(dst)
        raise HTTPException(400, "无法解析这个视频")

    # 已经有同一个视频就复用，不留副本 —— 否则列表里会出现两条一模一样的
    fp = _fingerprint(dst)
    for v in _list_videos():
        if v["fp"] == fp and os.path.realpath(v["path"]) != os.path.realpath(dst):
            os.unlink(dst)
            return dict(v, existed=True)

    return {"path": dst, "name": os.path.basename(dst), "duration": m["duration"],
            "width": m["width"], "height": m["height"],
            "quality": m["quality_score"], "note": m["recommendation"], "fp": fp}


@app.get("/source")
def source(path: str):
    """原视频，供页面预览。限制在允许的目录内，防止任意路径读取。"""
    rp = os.path.realpath(path)
    if not any(rp.startswith(os.path.realpath(d)) for d in VIDEO_DIRS):
        raise HTTPException(403, "路径不允许")
    if not os.path.exists(rp):
        raise HTTPException(404)
    return FileResponse(rp)


os.makedirs(OUT, exist_ok=True)
app.mount("/out", StaticFiles(directory=OUT), name="out")
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "web"), html=True), name="web")

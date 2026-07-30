"""Pipo AI 后端。直接调 ppai 管线，不复制逻辑。

三件事必须一起做，缺一个另外两个就是假的：
  * 持久化 —— 用户、视频归属、任务状态存 SQLite，重启不丢
  * 鉴权   —— 一人一条魔法链接，token 换 httpOnly cookie
  * 隔离   —— 每个用户只看得到自己上传的视频和自己的成片

特别注意：成片**不能**用 StaticFiles 挂载。挂载会绕过鉴权，
任何人猜到文件名就能下别人的视频。必须走带归属校验的路由。

任务放线程里异步跑：12 分钟视频要 10-30 秒，同步接口会把浏览器挂住。
进度只报阶段不报百分比 —— 管线内部没有进度回调，硬编码的百分比是假的。
"""
from __future__ import annotations

import base64
import glob
import hashlib
import json
import os
import subprocess
import threading
import time
import secrets
import traceback
import urllib.parse
from typing import Dict, List

from fastapi import (Depends, FastAPI, File, HTTPException, Request, Response,
                     UploadFile)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ppai import (audio, config, highlight, labels as L, motion, render, rerank,
                  stats, store)
from ppai.cli import probe

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
UPLOADS = os.path.join(ROOT, "uploads")
LABELS = os.path.join(ROOT, "labels")
ALLOWED_EXT = (".mp4", ".mov", ".m4v")

# 两条最小防护：没有它们，一次手滑（传 10GB）或者用久了
# （out/ 只增不减，实测一个 12 分钟视频产出 115MB）就能把服务弄挂。
MAX_UPLOAD_MB = int(os.environ.get("PIPO_MAX_UPLOAD_MB", "1024"))
KEEP_DAYS = float(os.environ.get("PIPO_KEEP_DAYS", "7"))
MAX_OUT_GB = float(os.environ.get("PIPO_MAX_OUT_GB", "10"))
CLEAN_EVERY_S = 1800
# 分析在总耗时里的占比，用来把两个阶段拼成一条进度。
# 实测：12 分钟素材分析 ~4s、渲染 ~20s；3.5 分钟素材 2s / 10s。都在 1:5 上下。
ANALYZE_SHARE = 0.18
STAR_N = 3                 # 结果里标几段「回合最长」
COOKIE = "pipo_sid"

app = FastAPI(title="Pipo AI")


# ── 鉴权 ──────────────────────────────────────────────────
def current_user(request: Request) -> Dict:
    """浏览器走 cookie，外部集成走 Bearer —— 同一个 token，两种取法。

    GPT Actions 和远程 MCP 连接器都发不了 cookie（不是浏览器），
    只支持 Authorization 头。不加这条，助手那边根本连不上。
    """
    tok = request.cookies.get(COOKIE, "")
    if not tok:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            tok = auth[7:].strip()
    u = store.user_by_token(tok)
    if not u:
        raise HTTPException(401, "未登录")
    return u


def user_dir(base: str, uid: str) -> str:
    d = os.path.join(base, uid)
    os.makedirs(d, exist_ok=True)
    return d


@app.get("/enter")
def enter(t: str):
    """魔法链接：/enter?t=TOKEN 换成 httpOnly cookie 后跳回首页。

    token 只在这一次出现在 URL 里，之后都在 cookie 中。
    httpOnly 让页面脚本读不到它，降低 XSS 的影响面。
    """
    u = store.user_by_token(t)
    if not u:
        return RedirectResponse("/login.html?bad=1", status_code=303)
    r = RedirectResponse("/", status_code=303)
    r.set_cookie(COOKIE, t, httponly=True, samesite="lax",
                 max_age=90 * 86400, path="/")
    return r


@app.post("/api/logout")
def logout():
    r = JSONResponse({"ok": True})
    r.delete_cookie(COOKIE, path="/")
    return r


@app.get("/api/me")
def me(u: Dict = Depends(current_user)):
    return {"id": u["id"], "name": u["name"], "is_admin": bool(u["is_admin"])}


# ── 数据模型 ──────────────────────────────────────────────
class Job(BaseModel):
    video: str = ""            # 单个 video id（MCP / skill 仍在用，保持兼容）
    videos: List[str] = []     # 多素材：当作一次训练课的若干段，合成一个成片
    themes: List[str] = ["auto"]
    top: int = 10
    ordered: bool = False      # 客户端已按用户意愿排好序，服务端不要再动

    def ids(self) -> List[str]:
        return self.videos or ([self.video] if self.video else [])


class Recut(BaseModel):
    video: str
    theme: str
    keep: List[int]


class Ingest(BaseModel):
    url: str
    name: str = ""


class Feedback(BaseModel):
    video: str
    start: float
    end: float
    verdict: str = "not_playing"


# 旧版质量提示的特征串，用来认出需要重算的老记录
_OLD_NOTES = ("视频质量良好", "分辨率低于 720p", "帧率低于 25fps",
              "方案假设为横屏固定机位", "无音轨 ——", "时长不足 2 分钟")


def shot_time(path: str) -> float:
    """拍摄时间，用于把多段素材按时间顺序接起来。

    优先容器里的 creation_time（手机直出的录像通常有），没有就退回文件 mtime。
    **mtime 不等于拍摄时间** —— 复制、下载、转码都会重置它。但同一批素材的
    相对先后通常还在，而错序的成片比「顺序可能不准」更糟。
    """
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
             "-of", "default=nw=1:nk=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        raw = p.stdout.decode().strip()
        if raw:
            from datetime import datetime
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        pass
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


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


# ── 任务 ──────────────────────────────────────────────────
_ANALYSIS: Dict[tuple, Dict] = {}
_ANALYSIS_INFLIGHT: Dict[tuple, threading.Event] = {}
_ANALYSIS_LOCK = threading.Lock()
_ANALYSIS_MAX = 8


def _analyze(video_paths: List[str], cfg: Dict, hcfg: Dict,
             step=None, pct=None) -> Dict:
    """检测 + 打分，出片前的全部重活。**结果按素材缓存**。

    抽出来是为了让「选主题」不再是盲选：主题预估和真正出片跑的是同一份
    分析，用户在挑主题时就已经把这步付过了，点「开始剪辑」只剩渲染。

    检测必须逐个视频做 —— 击球时间戳、运动曲线都是各自时间轴上的，
    拼起来算会在每个衔接处凭空造出一个巨大的空档。只有回合列表能合池，
    因为每个回合都带着自己的 src 和本地时间。

    缓存键是素材路径的**有序**元组：顺序变了合池后的时间轴也变了。
    视频本身不可变，所以不需要失效策略，只压容量。
    """
    key = tuple(video_paths)
    # 同一份素材的并发请求必须合并成一次计算。用户在素材间点来点去会连开
    # 好几个预估，而 motion 要把整段视频解一遍 —— 并发跑同一份素材会互相
    # 抢 CPU，每个都变慢，看起来就是「一直在估算」。第二个请求在这里等，
    # 不自己算。
    while True:
        with _ANALYSIS_LOCK:
            got = _ANALYSIS.get(key)
            if got is not None:
                return got
            ev = _ANALYSIS_INFLIGHT.get(key)
            if ev is None:
                ev = threading.Event()
                _ANALYSIS_INFLIGHT[key] = ev
                break                      # 这一轮由本线程负责算
        if not ev.wait(900):               # 等别人算完；超时就自己来
            with _ANALYSIS_LOCK:
                _ANALYSIS_INFLIGHT.pop(key, None)

    step = step or (lambda m: None)
    pct = pct or (lambda x: None)
    try:
        return _analyze_inner(video_paths, cfg, hcfg, step, pct, key, ev)
    except BaseException:
        with _ANALYSIS_LOCK:
            _ANALYSIS_INFLIGHT.pop(key, None)
        ev.set()
        raise


def _analyze_inner(video_paths, cfg, hcfg, step, pct, key, ev) -> Dict:
    import numpy as np

    multi = len(video_paths) > 1
    rc = cfg.get("rerank", {})
    model = (rerank.load(rc.get("model", rerank.MODEL_PATH))
             if rc.get("enabled") else None)
    total_src = sum(probe(p)["duration"] for p in video_paths) or 1.0
    done_src = [0.0]

    per, pool, gh, ga, base = [], [], [], [], 0.0
    for n, vp in enumerate(video_paths, 1):
        tag = "（%d/%d）" % (n, len(video_paths)) if multi else ""

        # 一段素材内部也要推进度：整段只在结束时跳一次的话，
        # 12 分钟的录像会有几十秒完全没有反馈 —— 正是要解决的问题。
        def sub(frac, _base=done_src[0], _d=0.0):
            pct(ANALYZE_SHARE * 100 * (_base + _d * frac) / total_src)

        dur_guess = probe(vp)["duration"]
        step("读取音轨" + tag)
        pcm = audio.extract_pcm(vp, cfg["audio"]["sr"])
        hits, env, _, fr = audio.detect_hits(pcm, cfg["audio"])
        sub(0.40, _d=dur_guess)
        if model is not None and len(hits):
            step("重排候选" + tag)
            hits = rerank.apply(vp, hits, model, rc.get("keep_ratio", 0.6))
        sub(0.60, _d=dur_guess)
        amps = audio.hit_amplitudes(hits, env, fr)
        step("分析画面运动" + tag)
        m_t, m_v = motion.motion_curve(vp, cfg["motion"])
        rs = highlight.score(highlight.rallies(hits, hcfg, amps), m_t, m_v, hcfg)
        for r in rs:
            r["src"] = vp
        meta = probe(vp)
        per.append({"path": vp, "hits": hits, "amps": amps,
                    "duration": meta["duration"]})
        pool.extend(rs)
        # 统计口径要的是「整堂课」，所以把各段时间轴接成一条：
        # 不加偏移，第二段的时间戳会落回第一段里面。
        gh.append(np.asarray(hits) + base)
        ga.append(np.asarray(amps))
        base += meta["duration"]
        done_src[0] += meta["duration"]
        pct(ANALYZE_SHARE * 100 * done_src[0] / total_src)

    hits_all = np.concatenate(gh) if gh else np.zeros(0)
    amps_all = np.concatenate(ga) if ga else np.zeros(0)
    vk = highlight.video_kind(pool, base, hcfg)
    summary = stats.summarize(pool, hits_all, amps_all, base, vk,
                              busy_gap=hcfg.get("busy_gap_s", 0.3))
    summary["source_count"] = len(video_paths)
    summary["sources"] = [os.path.basename(p) for p in video_paths]
    # 完整版的副标题要靠它解释「为什么比有效打球长得多」
    summary["trim_pad"] = hcfg.get("trim_pad_s", 0.5)
    res = {"per": per, "pool": pool, "summary": summary, "vk": vk, "total": base,
           "canvas": render.pick_canvas(
               video_paths, {p["path"]: p["duration"] for p in per})}
    with _ANALYSIS_LOCK:
        _ANALYSIS[key] = res
        while len(_ANALYSIS) > _ANALYSIS_MAX:
            _ANALYSIS.pop(next(iter(_ANALYSIS)))
        _ANALYSIS_INFLIGHT.pop(key, None)
    ev.set()
    return res


def _pick_for(kind: str, an: Dict, hcfg: Dict) -> List[Dict]:
    """某个主题会剪出哪些片段。**不做任何渲染** —— 这正是预估能便宜的原因。"""
    if kind == "trim":
        out = []
        for p in an["per"]:
            for s_ in highlight.trim_idle(p["hits"], p["duration"], hcfg,
                                          amps=p["amps"]):
                s_["src"] = p["path"]
                s_["id"] = len(out) + 1
                out.append(s_)
        return out
    if kind == "spot":
        return highlight.select(highlight.rank(an["pool"], kind, hcfg), hcfg,
                                total_s=hcfg.get("spot_seconds", 45))
    return highlight.select(highlight.rank(an["pool"], kind, hcfg), hcfg)


def _order(video_paths: List[str], ordered: bool) -> List[str]:
    if len(video_paths) > 1 and not ordered:
        # 默认按拍摄时间接续，而不是按点选顺序 —— 多选时点击顺序是随意的，
        # 而成片的时间线不该是随意的。网页端会自己排好并置 ordered=True，
        # 这时服务端不能再排一次，否则用户的调整会被悄悄覆盖。
        return sorted(video_paths, key=shot_time)
    return video_paths


def _run(jid: str, uid: str, video_paths: List[str], req: Job) -> None:
    def step(msg: str) -> None:
        store.update_job(jid, stage=msg)
    try:
        cfg = config.load()
        hcfg = dict(cfg["highlight"])
        hcfg["top_n"] = req.top
        video_paths = _order(video_paths, req.ordered)

        def pct(x):
            # 任务跑完前 payload 里只有进度，收尾时会被结果整体覆盖，直接整写
            store.update_job(jid, payload={"percent": round(min(99.0, x), 1)})

        # 用户挑主题时已经跑过分析了，这里直接命中缓存，只剩渲染
        an = _analyze(video_paths, cfg, hcfg, step=step, pct=pct)
        per, pool = an["per"], an["pool"]
        summary, vk, canvas = an["summary"], an["vk"], an["canvas"]
        pct(ANALYZE_SHARE * 100)

        kinds = req.themes
        if kinds == ["auto"]:
            # 完整版打底（一个球都不漏），再附一条几十秒的精华。
            # 两者不是二选一，是同一份素材的两种粒度：精华当场看，完整版存档。
            kinds = ["trim", "spot"]

        odir = user_dir(OUT, uid)
        stem = os.path.splitext(os.path.basename(video_paths[0]))[0][:40]
        if len(video_paths) > 1:
            stem = "%s_+%d" % (stem[:32], len(video_paths) - 1)
        results, used = [], []
        real = [k for k in kinds if k not in ("trim", "spot")]
        for kind in kinds:
            step("剪辑：%s" % highlight.RANKERS.get(kind, kind).split(" ")[0])
            picked = _pick_for(kind, an, hcfg)
            if kind not in ("trim", "spot") and len(real) > 1:
                picked = highlight.dedupe(picked, used, hcfg)
            # trim 和 spot 不参与去重：trim 覆盖了所有回合，拿它当「已用」
            # 会把精华整个去成空；而这两者本来就是同一批内容的不同粒度，
            # 去重要防的是「两个主题剪出同一段」，不是这种情况。
            if kind not in ("trim", "spot"):
                used.extend(picked)
            if not picked:
                results.append({"theme": kind, "empty": True})
                continue
            # 一屏几十个时间码等于把数据倒给用户让他自己找。标出回合最长的几段 ——
            # 用「拍数最多」这个能解释的事实，而不是编一个「精彩度」分数。
            star = {id(x) for x in sorted(
                picked, key=lambda r: (-r.get("hit_count", 0), -r.get("power", 0))
            )[:STAR_N]}
            for x in picked:
                x["star"] = id(x) in star
            # 烧进画面的一行说明。**不写拍数** —— 检测到的瞬态里混着台面弹跳，
            # 实测约为真实挥拍数的 2.7 倍（96 个回合 / 5 个视频），写出来是撒谎。
            # 相对排序是可靠的（Spearman 0.79），所以序号可以留。
            if cfg["render"].get("caption", True):
                for k, x in enumerate(picked, 1):
                    x["label"] = "第 %d 回合" % k
            seg_dir = "%s_hl_%s" % (stem, kind)
            ki = kinds.index(kind)
            label = highlight.RANKERS.get(kind, kind).split(" ")[0]

            def prog(i, n, _k=ki, _l=label):
                step("剪辑：%s %d/%d" % (_l, i, n))
                base_p = ANALYZE_SHARE + (1 - ANALYZE_SHARE) * _k / len(kinds)
                pct((base_p + (1 - ANALYZE_SHARE) / len(kinds) * i / n) * 100)

            parts = render.cut(video_paths[0], picked, os.path.join(odir, seg_dir),
                               cfg["render"], canvas=canvas, on_progress=prog)
            final = render.concat(parts, os.path.join(odir, "%s_%s.mp4" % (stem, kind)))
            off = 0.0
            for s_, part in zip(picked, parts):
                s_["url"] = "/media/%s/%s" % (seg_dir, os.path.basename(part))
                s_["offset"] = round(off, 2)
                off += s_["duration"]
            th = next((t for t in highlight.THEMES if t["id"] == kind), {})
            results.append({
                "theme": kind,
                "name": th.get("name", kind),
                "name_en": th.get("name_en", kind),
                "url": "/media/" + os.path.basename(final),
                "seconds": round(sum(s["duration"] for s in picked), 1),
                "clips": len(picked),
                "segments": picked,
            })
        store.update_job(jid, state="done", stage="完成",
                         payload={"results": results, "stats": summary, "kind": vk})
    except Exception as e:
        traceback.print_exc()
        store.update_job(jid, state="error", stage="失败", payload={"error": str(e)})


# ── 接口 ──────────────────────────────────────────────────
@app.get("/api/videos")
def videos(u: Dict = Depends(current_user)):
    out = []
    for v in store.list_videos(u["id"]):
        if not os.path.exists(v["path"]):     # 文件被清理掉了就不列
            continue
        # 这一列是后加的，老记录是空的 —— 首次列出时补齐，之后就不用再探了
        if not v.get("shot_at"):
            v["shot_at"] = shot_time(v["path"])
            store.set_shot_at(v["id"], v["shot_at"])
        # 质量提示是上传时算好存下来的，改了文案之后老记录还是旧话术。
        # 重新 probe 一次并存回，只发生一次。不自己重算是为了避免
        # 判断逻辑在两处各写一遍然后慢慢跑偏。
        if any(k in (v.get("note") or "") for k in _OLD_NOTES):
            m = probe(v["path"])
            store.set_note(v["id"], m["recommendation"], m["recommendation_en"],
                           m["quality_score"])
            v["note"], v["note_en"] = m["recommendation"], m["recommendation_en"]
            v["quality"] = m["quality_score"]
        d = {k: v[k] for k in
             ("id", "name", "duration", "width", "height", "quality",
              "note", "note_en")}
        d["shot_at"] = v["shot_at"]
        out.append(d)
    return out


class Est(BaseModel):
    video: str = ""
    videos: List[str] = []
    ordered: bool = False
    top: int = 10          # 必须和 Job.top 一致，否则预估的段数和实际出片对不上

    def ids(self) -> List[str]:
        return self.videos or ([self.video] if self.video else [])


@app.post("/api/estimate")
def estimate(req: Est, u: Dict = Depends(current_user)):
    """每个主题会剪出多少段、多长 —— 不渲染，只跑分析后做一次挑选。

    这是为了让选主题不再是盲选：原来用户想比较两个主题得各等一分钟出片，
    现在分析跑一次（结果缓存），七个主题的预估一起给出，
    之后真正出片直接命中缓存、只剩渲染。
    """
    ids = req.ids()
    if not ids:
        raise HTTPException(400, "没有选择视频")
    vs = [store.get_video(u["id"], i) for i in ids]
    if any(v is None for v in vs):
        raise HTTPException(404, "视频不存在")
    paths = _order([v["path"] for v in vs], req.ordered)
    for p in paths:
        if not os.path.exists(p):
            raise HTTPException(404, "素材文件已不在")
    cfg = config.load()
    hcfg = dict(cfg["highlight"])
    hcfg["top_n"] = req.top
    an = _analyze(paths, cfg, hcfg)
    out = {}
    for th in highlight.THEMES:
        k = th["id"]
        try:
            picked = _pick_for(k, an, hcfg)
        except Exception:
            continue
        out[k] = {"clips": len(picked),
                  "seconds": round(sum(x["duration"] for x in picked), 1)}
    # 「自动」= 完整版 + 精华，预估上也按这个口径给
    if "trim" in out and "spot" in out:
        out["auto"] = {"clips": out["trim"]["clips"] + out["spot"]["clips"],
                       "seconds": round(out["trim"]["seconds"]
                                        + out["spot"]["seconds"], 1),
                       "parts": [out["spot"], out["trim"]]}
    return {"themes": out, "stats": an["summary"]}


@app.get("/api/themes")
def themes():
    """不给前端返回 trim —— 它已经是「自动」的行为，再单列一个 chip 就是同一件事
    出现两次。THEMES 里仍然保留 trim，出片结果要靠它取名称和说明。"""
    return [t for t in highlight.THEMES if t["id"] not in ("trim", "spot")]


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...),
                 u: Dict = Depends(current_user)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "只支持 %s" % "/".join(ALLOWED_EXT))
    limit = MAX_UPLOAD_MB << 20
    # Content-Length 先挡一道（快速拒绝），但它是客户端给的、可以撒谎，
    # 所以边写边数才是真正的防线。
    try:
        if int(request.headers.get("content-length") or 0) > limit * 1.05:
            raise HTTPException(413, "文件超过 %d MB 上限" % MAX_UPLOAD_MB)
    except ValueError:
        pass

    udir = user_dir(UPLOADS, u["id"])
    safe = os.path.basename(file.filename or "video" + ext).replace("/", "_")
    dst = os.path.join(udir, safe)
    n = 1
    while os.path.exists(dst):
        st, e = os.path.splitext(safe)
        dst = os.path.join(udir, "%s_%d%s" % (st, n, e)); n += 1

    written = 0
    with open(dst, "wb") as fh:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > limit:
                fh.close(); os.unlink(dst)
                raise HTTPException(413, "文件超过 %d MB 上限" % MAX_UPLOAD_MB)
            fh.write(chunk)
    try:
        m = probe(dst)
    except Exception:
        os.unlink(dst)
        raise HTTPException(400, "无法解析这个视频")

    rec = store.add_video(u["id"], {
        "path": dst, "name": os.path.basename(dst), "fp": _fingerprint(dst),
        "duration": m["duration"], "width": m["width"], "height": m["height"],
        "quality": m["quality_score"], "note": m["recommendation"],
        "note_en": m.get("recommendation_en", "")})
    if rec["path"] != dst:            # 同一用户重复上传，复用旧记录
        os.unlink(dst)
    return {k: rec[k] for k in ("id", "name", "duration", "width", "height",
                                "quality", "note", "note_en")}


@app.post("/api/ingest")
def ingest(req: Ingest, u: Dict = Depends(current_user)):
    """按 URL 拉取视频。外部系统（助手插件、小程序、脚本）驱动的入口 ——
    表单上传只有浏览器能用。

    边下边数字节：Content-Length 是对方给的，可以撒谎，也可能根本不给。
    只允许 http(s)，且不解析到内网地址 —— 否则这是个 SSRF 洞，
    能拿它探测同机的其它服务。
    """
    import ipaddress
    import socket
    import urllib.parse
    import urllib.request

    p = urllib.parse.urlparse(req.url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise HTTPException(400, "只支持 http/https 链接")
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except OSError:
        raise HTTPException(400, "域名无法解析")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast):
            raise HTTPException(400, "不允许内网地址")

    ext = os.path.splitext(urllib.parse.unquote(p.path))[1].lower()
    if ext not in ALLOWED_EXT:
        ext = ".mp4"
    udir = user_dir(UPLOADS, u["id"])
    base = (req.name or os.path.basename(p.path) or "video")[:60]
    if not base.lower().endswith(ALLOWED_EXT):
        base += ext
    dst = os.path.join(udir, os.path.basename(base).replace("/", "_"))
    n = 1
    while os.path.exists(dst):
        st, e = os.path.splitext(dst)
        dst = "%s_%d%s" % (st, n, e); n += 1

    limit = MAX_UPLOAD_MB << 20
    written = 0
    try:
        rq = urllib.request.Request(req.url, headers={"User-Agent": "PipoAI/1.0"})
        with urllib.request.urlopen(rq, timeout=30) as r, open(dst, "wb") as fh:
            while chunk := r.read(1 << 20):
                written += len(chunk)
                if written > limit:
                    fh.close(); os.unlink(dst)
                    raise HTTPException(413, "文件超过 %d MB 上限" % MAX_UPLOAD_MB)
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(dst):
            os.unlink(dst)
        raise HTTPException(400, "拉取失败: %s" % str(e)[:120])

    try:
        m = probe(dst)
    except Exception:
        os.unlink(dst)
        raise HTTPException(400, "无法解析这个视频")
    rec = store.add_video(u["id"], {
        "path": dst, "name": os.path.basename(dst), "fp": _fingerprint(dst),
        "duration": m["duration"], "width": m["width"], "height": m["height"],
        "quality": m["quality_score"], "note": m["recommendation"],
        "note_en": m.get("recommendation_en", "")})
    if rec["path"] != dst:
        os.unlink(dst)
    return {k: rec[k] for k in ("id", "name", "duration", "width", "height",
                                "quality", "note", "note_en")}


@app.post("/api/jobs")
def create(req: Job, u: Dict = Depends(current_user)):
    ids = req.ids()
    if not ids:
        raise HTTPException(400, "没有选择视频")
    vs = [store.get_video(u["id"], i) for i in ids]
    if any(v is None for v in vs):
        raise HTTPException(404, "视频不存在")     # 含别人的视频时也走这里
    jid = store.create_job(u["id"], vs[0]["id"])
    threading.Thread(target=_run, args=(jid, u["id"], [v["path"] for v in vs], req),
                     daemon=True).start()
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str, u: Dict = Depends(current_user)):
    j = store.get_job(u["id"], jid)
    if not j:
        raise HTTPException(404, "任务不存在")
    return JSONResponse(j)


@app.post("/api/recut")
def recut(r: Recut, u: Dict = Depends(current_user)):
    """删除片段后重新拼接。

    删除和反馈是同一动作的两面：用户说「这段不该在这儿」，
    既要立刻从成片拿掉，也要记成负例。拆成两个按钮既让用户困惑，
    也会漏掉大部分反馈。

    重拼走 render.cut 重切而非直接拼现有片段 —— 淡出烘在最后一段里，
    删掉末段后直接拼会丢淡出。
    """
    v = store.get_video(u["id"], r.video)
    if not v:
        raise HTTPException(404, "视频不存在")
    j = store.last_done_job(u["id"], v["id"])
    if not j:
        raise HTTPException(404, "没有可重剪的任务")
    res = next((x for x in j.get("results", []) if x.get("theme") == r.theme), None)
    if not res or res.get("empty"):
        raise HTTPException(404, "没有这个主题的结果")

    keep = [s for s in res["segments"] if s["id"] in set(r.keep)]
    if not keep:
        raise HTTPException(400, "至少保留一个片段")
    cfg = config.load()
    odir = user_dir(OUT, u["id"])
    stem = os.path.splitext(os.path.basename(v["path"]))[0][:40]
    seg_dir = "%s_hl_%s" % (stem, r.theme)
    parts = render.cut(v["path"], keep, os.path.join(odir, seg_dir), cfg["render"])
    final = render.concat(parts, os.path.join(odir, "%s_%s.mp4" % (stem, r.theme)))
    off = 0.0
    for s_, part in zip(keep, parts):
        s_["url"] = "/media/%s/%s" % (seg_dir, os.path.basename(part))
        s_["offset"] = round(off, 2); off += s_["duration"]
    res["segments"] = keep; res["clips"] = len(keep); res["seconds"] = round(off, 1)
    store.update_job(j["id"], payload={"results": j["results"],
                                       "stats": j.get("stats"), "kind": j.get("kind")})
    # 加时间戳绕过浏览器缓存 —— 文件名没变，不加的话播放器还放旧的
    return {"url": "/media/%s?v=%d" % (os.path.basename(final), int(time.time())),
            "clips": len(keep), "seconds": round(off, 1), "segments": keep}


@app.post("/api/feedback")
def feedback(fb: Feedback, u: Dict = Depends(current_user)):
    """用户删片段 = 声明这段没有有效击球 = 可信负例。数据闭环的入口。

    写进 negative_ranges 而非 complete_ranges：后者零击球会被判为标注遗漏
    并跳过（那条保护是为了挡住「只标捡球没标击球」的情况）。
    """
    v = store.get_video(u["id"], fb.video)
    if not v:
        raise HTTPException(404, "视频不存在")
    if fb.end - fb.start <= 0.1:
        raise HTTPException(400, "区间太短")
    path = L.path_for(v["path"], LABELS)
    lab = L.load(path) if os.path.exists(path) else L.empty(v["path"], v["duration"])
    lab["negative_ranges"] = list(lab.get("negative_ranges") or []) + [[fb.start, fb.end]]
    L.save(lab, path)
    # 立刻固化这段的嵌入 —— 原片会按生命周期规则删除，晚了就没得算了。
    # 放后台是因为要跑一次 PANNs，几秒钟，不该让用户的点击等着。
    threading.Thread(target=_freeze_quietly, args=(v["path"], path), daemon=True).start()
    return {"ok": True, "ranges": len(lab["negative_ranges"])}


def _freeze_quietly(video: str, label_path: str) -> None:
    try:
        rerank.freeze(video, label_path, config.load())
    except Exception:
        traceback.print_exc()


@app.get("/api/feedback")
def feedback_summary(u: Dict = Depends(current_user)):
    mine = {os.path.realpath(v["path"]) for v in store.list_videos(u["id"])}
    out, n, sec = [], 0, 0.0
    for f in sorted(glob.glob(os.path.join(LABELS, "*.json"))):
        try:
            lab = L.load(f)
        except Exception:
            continue
        if os.path.realpath(lab["video"]) not in mine:   # 只统计自己的
            continue
        r = lab.get("negative_ranges") or []
        if not r:
            continue
        t = sum(b - a for a, b in r)
        n += len(r); sec += t
        out.append({"video": os.path.basename(lab["video"]),
                    "ranges": len(r), "seconds": round(t, 1)})
    return {"videos": out, "total_ranges": n, "total_seconds": round(sec, 1)}


@app.get("/media/{rest:path}")
def media(rest: str, u: Dict = Depends(current_user)):
    """成片和片段。**不能用 StaticFiles 挂载** —— 那会绕过鉴权，
    任何人猜到文件名就能下别人的视频。"""
    base = os.path.realpath(user_dir(OUT, u["id"]))
    p = os.path.realpath(os.path.join(base, rest))
    if not p.startswith(base + os.sep) or not os.path.isfile(p):
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/source")
def source(id: str, u: Dict = Depends(current_user)):
    """原视频预览。按 video id 取 —— 归属校验和取数据是同一次查询，
    不给「忘了检查」留机会。"""
    v = store.get_video(u["id"], id)
    if not v or not os.path.exists(v["path"]):
        raise HTTPException(404)
    return FileResponse(v["path"])


@app.get("/api/thumb")
def thumb(id: str, u: Dict = Depends(current_user)):
    """缩略图。生成一次存盘复用 —— 每次列表都抽帧的话，
    五个视频就是五次 ffmpeg，侧栏会肉眼可见地卡。

    取 8% 处而不是首帧：录像开头常常是走向球台、镜头还在晃，
    首帧多半是黑的或者糊的，作为封面认不出是哪一段。
    """
    v = store.get_video(u["id"], id)
    if not v or not os.path.exists(v["path"]):
        raise HTTPException(404)
    d = user_dir(os.path.join(ROOT, "thumbs"), u["id"])
    dst = os.path.join(d, "%s.jpg" % id)
    if not os.path.exists(dst):
        at = max(0.5, (v["duration"] or 10) * 0.08)
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", "%.2f" % at, "-i", v["path"],
             "-frames:v", "1", "-vf", "scale=160:-2", "-q:v", "5", dst],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if p.returncode != 0 or not os.path.exists(dst):
            raise HTTPException(500, "缩略图生成失败")
    # 缩略图不会变（视频是不可变的），让浏览器长期缓存，别每次刷新都回源
    return FileResponse(dst, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=604800"})


@app.get("/api/storage")
def storage(u: Dict = Depends(current_user)):
    d = os.path.join(OUT, u["id"])
    total = 0
    if os.path.isdir(d):
        for root, _, fs in os.walk(d):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in fs
                         if os.path.exists(os.path.join(root, f)))
    return {"out_mb": round(total / (1 << 20), 1), "cap_gb": MAX_OUT_GB,
            "keep_days": KEEP_DAYS, "max_upload_mb": MAX_UPLOAD_MB}


# ── OAuth（MCP 连接器用）──────────────────────────────────
# 为什么要这个：远程连接器不能让用户手工粘 token —— 又难用又容易泄露。
# MCP 规范里的做法是助手把用户弹到**我们的**授权页，同意后 token 自动回传。
#
# 注意：这不是「用 ChatGPT/Claude 账号登录」—— 两家都没有面向第三方的
# 公开身份服务。身份始终是我们自己的（邀请码），助手只负责跑授权流程。
# 对用户来说效果一样：点一下同意，全程不碰 token。
@app.get("/.well-known/oauth-protected-resource")
@app.get("/.well-known/oauth-protected-resource/mcp")
def oauth_prm(request: Request):
    base = str(request.base_url).rstrip("/")
    return {"resource": base + "/mcp", "authorization_servers": [base]}


@app.get("/.well-known/oauth-authorization-server")
def oauth_meta(request: Request):
    base = str(request.base_url).rstrip("/")
    return {
        "issuer": base,
        "authorization_endpoint": base + "/oauth/authorize",
        "token_endpoint": base + "/oauth/token",
        "registration_endpoint": base + "/oauth/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
    }


@app.post("/oauth/register")
async def oauth_register(request: Request):
    """动态客户端注册（RFC 7591）。MCP 客户端会自己来注册，
    我们不做审核 —— 真正的门槛是后面那一步用户必须输入邀请码。"""
    body = await request.json()
    return JSONResponse({
        "client_id": "mcp-" + secrets.token_hex(8),
        "client_name": body.get("client_name", "MCP Client"),
        "redirect_uris": body.get("redirect_uris", []),
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code"],
        "response_types": ["code"],
    }, status_code=201)


@app.get("/oauth/authorize")
def oauth_authorize(request: Request, client_id: str = "", redirect_uri: str = "",
                    state: str = "", code_challenge: str = "",
                    code_challenge_method: str = "S256",
                    scope: str = "", response_type: str = "code"):
    if code_challenge_method != "S256" or not code_challenge:
        raise HTTPException(400, "需要 PKCE (S256)")
    # 参数原样留在 URL 里，页面自己 location.search 读 —— 不用绕 header
    return FileResponse(os.path.join(ROOT, "web", "authorize.html"))


@app.post("/oauth/approve")
async def oauth_approve(request: Request):
    """用户在授权页点「同意」。身份来自邀请码或已有 cookie。"""
    b = await request.json()
    tok = b.get("token") or request.cookies.get(COOKIE, "")
    u = store.user_by_token(tok)
    if not u:
        raise HTTPException(401, "邀请码无效")
    ru = b.get("redirect_uri") or ""
    if not ru.startswith(("http://localhost", "http://127.0.0.1", "https://")):
        raise HTTPException(400, "回调地址不合法")
    code = store.put_code(u["id"], b.get("client_id", ""), ru,
                          b.get("code_challenge", ""))
    sep = "&" if "?" in ru else "?"
    q = {"code": code}
    if b.get("state"):
        q["state"] = b["state"]
    return {"redirect": ru + sep + urllib.parse.urlencode(q)}


@app.post("/oauth/token")
async def oauth_token(request: Request):
    form = await request.form()
    if form.get("grant_type") != "authorization_code":
        raise HTTPException(400, "unsupported_grant_type")
    rec = store.take_code(form.get("code") or "")
    if not rec:
        raise HTTPException(400, "授权码无效或已过期")
    # PKCE 校验：没有它，截获授权码的人就能换到 token
    verifier = form.get("code_verifier") or ""
    digest = hashlib.sha256(verifier.encode()).digest()
    if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != rec["challenge"]:
        raise HTTPException(400, "PKCE 校验失败")
    u = next((x for x in store.list_users() if x["id"] == rec["user_id"]), None)
    if not u:
        raise HTTPException(400, "用户不存在")
    return {"access_token": u["token"], "token_type": "Bearer",
            "scope": "pipo", "expires_in": 90 * 86400}


# ── MCP over HTTP ─────────────────────────────────────────
# 远程连接器（claude.ai、ChatGPT）用 HTTP 传输，用户粘个 URL 就能加，
# 不用装 Python 也不用改配置文件。mcp_server.py 那个 stdio 版只覆盖
# Claude Desktop / Claude Code 这类本地场景，两个都要留。
MCP_TOOLS = [
    {"name": "pipo_upload_link",
     "annotations": {"title": "Get upload link", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True,
                     "openWorldHint": False},
     "description": ("当用户想剪一段本地/手机里的录像时调用。助手无法接收几百 MB 的"
                     "视频文件，所以返回一个已带登录的上传链接。把 url 和 instructions "
                     "原样告诉用户，提示传完回来说一声，然后用 pipo_list_videos 取新视频。"),
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "pipo_list_videos",
     "annotations": {"title": "List videos", "readOnlyHint": True,
                     "destructiveHint": False, "idempotentHint": True,
                     "openWorldHint": False},
     "description": "列出当前用户已上传的乒乓球训练录像，返回 id / 名称 / 时长。",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "pipo_add_video",
     "annotations": {"title": "Add video by URL", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": False,
                     "openWorldHint": True},
     "description": "按公开 http/https 直链添加一段录像，返回 video_id。传不了本地文件。",
     "inputSchema": {"type": "object",
                     "properties": {"url": {"type": "string"},
                                    "name": {"type": "string"}},
                     "required": ["url"]}},
    {"name": "pipo_make_highlight",
     "annotations": {"title": "Make highlight reel", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": False,
                     "openWorldHint": False},
     "description": ("生成集锦并返回下载链接，通常十几秒到一分钟。theme 可选："
                     "auto（默认，等于完整版：去掉捡球和等待、一个球都不漏）、"
                     "best 训练集锦、longest 最长相持、"
                     "power 扣杀瞬间（单次声音峰值，最可靠）、"
                     "weak 失误合集、records 精彩瞬间。"),
     "inputSchema": {"type": "object",
                     "properties": {"video_id": {"type": "string"},
                                    "theme": {"type": "string", "default": "auto"},
                                    "top": {"type": "integer", "default": 10}},
                     "required": ["video_id"]}},
]


def _mcp_dispatch(name: str, args: Dict, u: Dict, base: str) -> Dict:
    if name == "pipo_upload_link":
        return {"url": "%s/enter?t=%s" % (base, u["token"]),
                "instructions": ("点开链接（已带登录，无需注册），把训练录像拖进左侧"
                                 "或点「选择文件」。支持 mp4/mov，单个最大 %d MB。"
                                 "传完回来说一声，我接着帮你剪。" % MAX_UPLOAD_MB)}
    if name == "pipo_list_videos":
        return {"videos": videos(u)}
    if name == "pipo_add_video":
        return ingest(Ingest(url=args["url"], name=args.get("name", "")), u)
    if name == "pipo_make_highlight":
        jid = store.create_job(u["id"], args["video_id"])
        v = store.get_video(u["id"], args["video_id"])
        if not v:
            raise HTTPException(404, "视频不存在")
        # 同步跑：MCP 调用方在等返回，异步反而要它自己轮询。
        # 一小时视频约 48 秒，在助手可接受的等待范围内。
        _run(jid, u["id"], [v["path"]],
             Job(video=args["video_id"], themes=[args.get("theme", "auto")],
                 top=int(args.get("top", 10))))
        j = store.get_job(u["id"], jid)
        if j["state"] != "done":
            raise HTTPException(500, j.get("error") or "任务失败")
        out = [{"theme": r["theme"], "name": r["name"], "clips": r["clips"],
                "seconds": r["seconds"], "url": base + r["url"]}
               for r in j.get("results", []) if not r.get("empty")]
        return {"results": out, "stats": j.get("stats", {})}
    raise HTTPException(400, "未知工具: %s" % name)


@app.post("/mcp")
async def mcp(request: Request):
    # 401 必须带 WWW-Authenticate 指向资源元数据，客户端才知道去哪儿授权。
    # 少了这个头，claude.ai 只会报「连接失败」而不会发起 OAuth。
    try:
        u = current_user(request)
    except HTTPException:
        base = str(request.base_url).rstrip("/")
        return JSONResponse(
            {"error": "unauthorized"}, status_code=401,
            headers={"WWW-Authenticate":
                     'Bearer resource_metadata="%s/.well-known/'
                     'oauth-protected-resource"' % base})
    return await _mcp_body(request, u)


async def _mcp_body(request: Request, u: Dict):
    """MCP Streamable HTTP 端点。用户在 claude.ai / ChatGPT 里
    粘贴 https://域名/mcp 并填 token 即可，无需本地安装。"""
    msg = await request.json()
    mid, method = msg.get("id"), msg.get("method")
    base = str(request.base_url).rstrip("/")

    if method == "initialize":
        r = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
             "serverInfo": {"name": "pipo-ai", "version": "0.1.0"}}
    elif method == "tools/list":
        r = {"tools": MCP_TOOLS}
    elif method == "tools/call":
        p = msg.get("params", {})
        try:
            res = _mcp_dispatch(p.get("name", ""), p.get("arguments") or {}, u, base)
            r = {"content": [{"type": "text",
                              "text": json.dumps(res, ensure_ascii=False)}]}
        except HTTPException as e:
            r = {"content": [{"type": "text", "text": "出错：%s" % e.detail}],
                 "isError": True}
        except Exception as e:
            traceback.print_exc()
            r = {"content": [{"type": "text", "text": "出错：%s" % e}],
                 "isError": True}
    elif method and method.startswith("notifications/"):
        return Response(status_code=202)          # 通知无需回复
    else:
        return JSONResponse({"jsonrpc": "2.0", "id": mid,
                             "error": {"code": -32601, "message": "method not found"}})
    return JSONResponse({"jsonrpc": "2.0", "id": mid, "result": r})


# ── 清理 ──────────────────────────────────────────────────
def _sweep() -> Dict:
    """清理 out/：先删过期的，还超容量就从最旧的继续删。

    只动 out/ —— labels/ 是训练数据、models/ 是模型、cache/ 是嵌入缓存、
    uploads/ 是用户原片，删了要重算或直接丢失，都不该被自动清理碰。
    """
    if not os.path.isdir(OUT):
        return {"removed": 0, "freed_mb": 0, "remaining_mb": 0}
    now = time.time()
    items = []
    for root, _, files in os.walk(OUT):
        for f in files:
            p = os.path.join(root, f)
            try:
                items.append([p, os.path.getsize(p), os.path.getmtime(p)])
            except OSError:
                pass

    def rm(p):
        try:
            os.unlink(p); return True
        except OSError:
            return False

    removed = freed = 0
    keep = []
    for p, sz, mt in items:
        if now - mt > KEEP_DAYS * 86400 and rm(p):
            removed += 1; freed += sz
        else:
            keep.append([p, sz, mt])

    cap = int(MAX_OUT_GB * (1 << 30))
    total = sum(x[1] for x in keep)
    for p, sz, mt in sorted(keep, key=lambda x: x[2]):
        if total <= cap:
            break
        if rm(p):
            removed += 1; freed += sz; total -= sz
    return {"removed": removed, "freed_mb": round(freed / (1 << 20), 1),
            "remaining_mb": round(total / (1 << 20), 1)}


def _sweeper():
    while True:
        try:
            r = _sweep()
            if r["removed"]:
                print("[cleanup] 删除 %d 项，释放 %.1f MB" % (r["removed"], r["freed_mb"]))
        except Exception:
            traceback.print_exc()
        time.sleep(CLEAN_EVERY_S)


# 未登录访问首页时跳登录页
@app.middleware("http")
async def gate(request: Request, call_next):
    if request.url.path in ("/", "/index.html") and not store.user_by_token(
            request.cookies.get(COOKIE, "")):
        return RedirectResponse("/login.html", status_code=303)
    return await call_next(request)


for _d in (OUT, UPLOADS, LABELS):
    os.makedirs(_d, exist_ok=True)
_n = store.orphan_running_jobs()
if _n:
    print("[startup] %d 个运行中的任务因重启被标记为中断" % _n)
threading.Thread(target=_sweeper, daemon=True).start()
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "web"), html=True), name="web")

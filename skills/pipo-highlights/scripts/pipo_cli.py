#!/usr/bin/env python3
"""Pipo AI 命令行客户端 —— 乒乓球训练录像自动剪集锦。

只用标准库，不依赖 Pipo 仓库：技能是独立安装的，装到用户机器上时
身边不会有那个 repo。所有逻辑都在服务端 HTTP API 里，这里只是薄封装。

两种连法，按顺序尝试：

  1. **本机桌面版**（推荐，完全离线）—— Pipo.app 启动时会把自己的端口写到
     ~/Library/Application Support/Pipo/port，这里自动读取。
     本地模式没有鉴权，不需要令牌，整个过程不联网。
  2. **远端服务** —— 设 PIPO_BASE_URL 和 PIPO_TOKEN。

环境变量：
    PIPO_BASE_URL   选填，显式指定服务地址（设了就不再找本机 app）
    PIPO_TOKEN      连远端服务时必填；连本机桌面版时不需要

用法：
    pipo_cli.py list
    pipo_cli.py upload <本地视频路径>
    pipo_cli.py cut <video_id> [--theme auto] [--top 10]
"""
from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

def _local_port() -> str:
    """读本机桌面版写下的端口。它每次启动随机取端口（固定端口会撞），
    所以约定写到数据目录的 port 文件里。"""
    for p in (os.path.expanduser("~/Library/Application Support/Pipo/port"),
              os.path.expanduser("~/.local/share/Pipo/port")):
        try:
            with open(p, encoding="utf-8") as f:
                v = f.read().strip()
            if v.isdigit():
                return v
        except OSError:
            continue
    return ""


def _resolve_base() -> str:
    env = os.environ.get("PIPO_BASE_URL")
    if env:
        return env.rstrip("/")
    port = _local_port()
    if port:
        return "http://127.0.0.1:%s" % port
    return "http://127.0.0.1:8020"


BASE = _resolve_base()
TOKEN = os.environ.get("PIPO_TOKEN", "")
LOCAL = BASE.startswith("http://127.0.0.1") and not os.environ.get("PIPO_BASE_URL")

THEMES = ("auto", "best", "longest", "power", "trim")


def die(msg: str) -> None:
    print("错误：%s" % msg, file=sys.stderr)
    raise SystemExit(1)


def call(method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json",
                 # 本机桌面版没有鉴权；带一个空 Bearer 反而会让远端服务
                 # 报 401 而不是「没给令牌」，所以没有令牌时干脆不发这个头
                 **({"Authorization": "Bearer %s" % TOKEN} if TOKEN else {})})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            pass
        if e.code == 401:
            die("令牌无效或已过期，检查 PIPO_TOKEN")
        die("%s %s: %s" % (e.code, e.reason, detail))
    except urllib.error.URLError as e:
        if LOCAL:
            die("连不上本机的 Pipo（%s）。先打开 Pipo 桌面版再试 —— "
                "它启动后会把端口写到 ~/Library/Application Support/Pipo/port。" % BASE)
        die("连不上 %s（%s）—— 服务没起来，或 PIPO_BASE_URL 不对" % (BASE, e.reason))


def upload(path: str) -> dict:
    """流式 multipart 上传。

    不能整个读进内存 —— 训练录像动辄几百 MB，read() 会直接把进程撑爆。
    手工拼 multipart 再分块 send，内存占用与文件大小无关。
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        die("文件不存在: %s" % path)
    if os.path.splitext(path)[1].lower() not in (".mp4", ".mov", ".m4v"):
        die("只支持 mp4 / mov / m4v")

    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "video/mp4"
    boundary = "----pipo" + os.urandom(8).hex()
    head = ("--%s\r\nContent-Disposition: form-data; name=\"file\"; "
            "filename=\"%s\"\r\nContent-Type: %s\r\n\r\n"
            % (boundary, name, ctype)).encode()
    tail = ("\r\n--%s--\r\n" % boundary).encode()
    size = os.path.getsize(path)

    u = urllib.parse.urlparse(BASE)
    Conn = (http.client.HTTPSConnection if u.scheme == "https"
            else http.client.HTTPConnection)
    conn = Conn(u.hostname, u.port, timeout=1800)
    conn.putrequest("POST", "/api/upload")
    conn.putheader("Content-Type", "multipart/form-data; boundary=" + boundary)
    conn.putheader("Content-Length", str(len(head) + size + len(tail)))
    conn.putheader("Authorization", "Bearer %s" % TOKEN)
    conn.endheaders()
    conn.send(head)
    sent = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            conn.send(chunk)
            sent += len(chunk)
            pct = sent * 100 // size
            print("\r  上传中 %3d%%" % pct, end="", file=sys.stderr, flush=True)
    conn.send(tail)
    print("\r  上传完成    ", file=sys.stderr)
    r = conn.getresponse()
    raw = r.read()
    if r.status != 200:
        detail = ""
        try:
            detail = json.loads(raw).get("detail", "")
        except Exception:
            pass
        die("上传失败 %d: %s" % (r.status, detail or raw[:120]))
    return json.loads(raw)


def cut(video_id: str, theme: str, top: int) -> dict:
    job = call("POST", "/api/jobs",
               {"video": video_id, "themes": [theme], "top": top})
    # 轮询而不是让请求挂住：一小时的录像要几十秒，
    # 中间任何一跳超时都会让调用方看到连接中断而不是结果。
    last = ""
    for _ in range(180):
        time.sleep(2)
        j = call("GET", "/api/jobs/%s" % job["id"])
        if j.get("stage") and j["stage"] != last:
            last = j["stage"]
            print("  %s" % last, file=sys.stderr)
        if j["state"] == "done":
            out = [{"theme": r["theme"], "name": r["name"], "clips": r["clips"],
                    "seconds": r["seconds"], "url": BASE + r["url"]}
                   for r in j.get("results", []) if not r.get("empty")]
            return {"results": out, "stats": j.get("stats", {})}
        if j["state"] == "error":
            die(j.get("error") or j.get("stage") or "任务失败")
    die("超时：任务超过 6 分钟仍未完成")


def main() -> None:
    ap = argparse.ArgumentParser(prog="pipo_cli.py", description="Pipo AI 集锦剪辑")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出已上传的录像")
    up = sub.add_parser("upload", help="上传本地视频")
    up.add_argument("path")
    ct = sub.add_parser("cut", help="生成集锦")
    ct.add_argument("video_id")
    ct.add_argument("--theme", default="auto", choices=THEMES)
    ct.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    # 只有连远端服务才需要令牌。本机桌面版是单用户、只监听 127.0.0.1，
    # 没有鉴权 —— 在这里硬性要求令牌会把「完全离线」这条路堵死。
    if not TOKEN and not LOCAL:
        die("连的是远端服务但没有设置 PIPO_TOKEN。"
            "要用本机桌面版的话，打开 Pipo 并去掉 PIPO_BASE_URL 即可。")

    if a.cmd == "list":
        out = call("GET", "/api/videos")
    elif a.cmd == "upload":
        out = upload(a.path)
    else:
        out = cut(a.video_id, a.theme, a.top)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

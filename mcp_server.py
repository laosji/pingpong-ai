"""Pipo AI 的 MCP server —— 让对话助手能驱动剪辑。

它是**薄封装**：所有逻辑都在 HTTP API 里，这里只负责把 MCP 的工具调用
转成 HTTP 请求。这样网页端和助手端永远不会行为分叉。

用法（MCP 客户端配置）：
    command: python
    args: ["/path/to/mcp_server.py"]
    env:
      PIPO_BASE_URL: "https://pipo.example.com"
      PIPO_TOKEN: "用户的魔法链接里那串 token"

**关于「视频怎么进来」这个前提问题**
本模块只接受**视频 URL**，因为 MCP 的工具参数是 JSON，传不了几百 MB 的
本地文件。所以助手这条路能不能成为主入口，取决于该助手能否把用户在对话里
上传的文件暴露成一个临时 URL。能，流程就通；不能，助手只适合做
「查询已处理的视频」这类轻交互，上传仍得走网页。
这一点在接任何一家平台之前都该先确认。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

BASE = os.environ.get("PIPO_BASE_URL", "http://127.0.0.1:8020").rstrip("/")
TOKEN = os.environ.get("PIPO_TOKEN", "")


def _call(method: str, path: str, body: Optional[Dict] = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json",
                 "Cookie": "pipo_sid=%s" % TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            pass
        raise RuntimeError("%s: %s" % (e.code, detail or e.reason))


TOOLS = [
    {
        "name": "pipo_upload_link",
        "description": (
            "当用户想剪一段**本地/手机里**的录像时调用这个。\n"
            "助手无法接收几百 MB 的视频文件，所以返回一个专属上传链接，"
            "让用户在浏览器里传。把返回的 url 和 instructions 原样告诉用户，"
            "并提示传完回来说一声，然后用 pipo_list_videos 取新视频。"),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "pipo_list_videos",
        "description": "列出当前用户已上传的乒乓球训练录像。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "pipo_add_video",
        "description": ("按 URL 添加一段乒乓球录像。只接受可公开访问的 http/https "
                        "直链，不能传本地文件。返回 video_id。"),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "视频直链"},
                "name": {"type": "string", "description": "可选，显示用的文件名"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "pipo_make_highlight",
        "description": (
            "为一段录像生成集锦并返回下载链接。整个过程通常十几秒到一分钟。\n"
            "theme 可选：auto（默认，自动判断素材类型选题）、best 训练集锦、"
            "longest 最长对拉、kill 最帅击球、power 最重扣杀、"
            "trim 完整版（只剪掉等待，保留所有球）、records 精彩瞬间。"),
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "theme": {"type": "string", "default": "auto"},
                "top": {"type": "integer", "default": 10,
                        "description": "取前几个片段，trim 主题下无效"},
            },
            "required": ["video_id"],
        },
    },
]


def _highlight(video_id: str, theme: str = "auto", top: int = 10) -> Dict:
    job = _call("POST", "/api/jobs",
                {"video": video_id, "themes": [theme], "top": top})
    # 轮询而不是让 HTTP 请求挂住：一小时视频要几十秒，
    # 中间任何一跳超时都会让调用方看到连接中断而不是结果。
    for _ in range(180):
        time.sleep(2)
        j = _call("GET", "/api/jobs/%s" % job["id"])
        if j["state"] == "done":
            out = []
            for r in j.get("results", []):
                if r.get("empty"):
                    continue
                out.append({"theme": r["theme"], "name": r["name"],
                            "clips": r["clips"], "seconds": r["seconds"],
                            "url": BASE + r["url"]})
            return {"results": out, "stats": j.get("stats", {})}
        if j["state"] == "error":
            raise RuntimeError(j.get("error") or j.get("stage") or "任务失败")
    raise RuntimeError("超时：任务超过 6 分钟仍未完成")


def dispatch(name: str, args: Dict) -> Any:
    if name == "pipo_upload_link":
        # 助手传不了本地大文件，只能把入口递给用户 —— 这是这类集成的固有边界，
        # 不是实现问题。与其让用户先自己找地方传再回来给链接，
        # 不如直接给一条点开就能用的专属地址。
        return {
            "url": "%s/enter?t=%s" % (BASE, TOKEN),
            "instructions": (
                "点开链接（已带登录，无需注册），把训练录像拖进左侧或点「选择文件」。"
                "支持 mp4/mov，单个最大 1GB。传完回来说一声，我接着帮你剪。"),
            "instructions_en": (
                "Open the link (already signed in), then drop your training video "
                "on the left or click Choose file. mp4/mov, up to 1GB. "
                "Tell me when it's uploaded and I'll take it from there."),
        }
    if name == "pipo_list_videos":
        return _call("GET", "/api/videos")
    if name == "pipo_add_video":
        return _call("POST", "/api/ingest",
                     {"url": args["url"], "name": args.get("name", "")})
    if name == "pipo_make_highlight":
        return _highlight(args["video_id"], args.get("theme", "auto"),
                          int(args.get("top", 10)))
    raise RuntimeError("未知工具: %s" % name)


# ── MCP over stdio ────────────────────────────────────────
def _send(obj: Dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid, method = msg.get("id"), msg.get("method")

        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "pipo-ai", "version": "0.1.0"}}})
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            p = msg.get("params", {})
            try:
                res = dispatch(p.get("name", ""), p.get("arguments") or {})
                _send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": json.dumps(res, ensure_ascii=False)}]}})
            except Exception as e:
                _send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "出错：%s" % e}],
                    "isError": True}})
        elif mid is not None:
            _send({"jsonrpc": "2.0", "id": mid,
                   "error": {"code": -32601, "message": "method not found"}})


if __name__ == "__main__":
    main()

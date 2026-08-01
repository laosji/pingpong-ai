"""Pipo 桌面版入口。

结构：本机起一个只监听 127.0.0.1 的 FastAPI，再用系统 WebKit 套一个原生窗口
指过去。界面、剪辑逻辑、主题全是现成的 —— 桌面版和网页版是同一套代码，
差别只有 PIPO_LOCAL=1 打开的那几处（见 server.py 顶部）。

为什么不重写成纯原生：剪辑逻辑在 Python 里（numpy + onnxruntime + ffmpeg），
重写一遍 UI 只是换个壳，却要把两套界面长期同步。用 WebKit 意味着不打包
Chromium，壳本身只有几 MB。

三件必须由原生侧做的事，网页做不到：
  * 选文件  —— 浏览器的 <input type=file> 拿不到真实路径，只能拿到文件内容。
                而我们要的恰恰是路径（不拷贝，见 /api/local/add）。
  * 在访达中显示成片
  * 端口     —— 固定端口会和别的程序撞，让系统分配一个空闲的
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

APP_NAME = "Pipo"
VIDEO_EXT = ("mp4", "mov", "m4v", "MP4", "MOV", "M4V")


def data_dir() -> str:
    """用户数据放 ~/Library/Application Support/Pipo。

    不能放应用包里 —— 那是只读的（而且换版本会被整个替换掉）。
    """
    d = os.path.expanduser("~/Library/Application Support/%s" % APP_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def bundled() -> str:
    """打包后资源在 sys._MEIPASS，开发时就是脚本所在目录。"""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def port_file() -> str:
    return os.path.join(data_dir(), "port")


def write_port(port: int) -> None:
    """把端口写到一个固定位置，让外部程序（LLM 技能、脚本）找得到本机的 Pipo。

    端口是每次启动随机取的 —— 固定端口会和别的程序撞。但随机端口意味着
    别人无从知道，所以要留一个约定的落点。

    **只写 127.0.0.1 的端口，不写任何令牌**：本地模式没有鉴权，
    而这个文件本身受用户目录的权限保护。
    """
    try:
        with open(port_file(), "w", encoding="utf-8") as f:
            f.write(str(port))
    except OSError:
        pass          # 写不了不该让 app 起不来，只是技能连不上


def clear_port() -> None:
    try:
        os.remove(port_file())
    except OSError:
        pass


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def serve(port: int) -> None:
    import uvicorn

    import server
    # log_level=warning：正常请求日志对用户没有意义，出错还是会打出来
    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")


def wait_ready(port: int, timeout: float = 30.0) -> bool:
    """等后端起来。不能起完就开窗口 —— 会白屏，用户以为坏了。"""
    url = "http://127.0.0.1:%d/api/me" % port
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except urllib.error.HTTPError:
            return True          # 有响应就算起来了，401 之类也行
        except Exception:
            time.sleep(0.15)
    return False


class Bridge:
    """暴露给页面的原生能力。页面里通过 window.pywebview.api.xxx() 调。"""

    def __init__(self):
        self.window = None

    def pick_videos(self):
        """系统文件选择器，多选，返回**绝对路径**列表。

        浏览器的文件输入框只能给到文件内容，拿不到路径 —— 而本地版
        整个不拷贝的设计就建立在能拿到路径上。这是必须走原生的原因。
        """
        import webview

        got = self.window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=True,
            file_types=("视频 (%s)" % ";".join("*.%s" % e for e in VIDEO_EXT),))
        return list(got or [])

    def reveal(self, path: str):
        """在访达里选中这个文件。成片剪完之后用户下一步就是找到它。"""
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.exists(path):
            subprocess.run(["open", "-R", path], check=False)
            return True
        return False

    def open_path(self, path: str):
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.exists(path):
            subprocess.run(["open", path], check=False)
            return True
        return False

    def data_dir(self):
        return data_dir()


def main() -> int:
    import webview

    root = bundled()
    # 必须在 import server 之前设好：server 和 ppai.paths 在模块加载时
    # 就把路径算成常量了，之后再改环境变量不起作用。
    os.environ["PIPO_LOCAL"] = "1"
    os.environ.setdefault("PIPO_DATA_DIR", data_dir())
    # 打包后 ffmpeg/ffprobe 在应用包里，得让子进程找得到
    os.environ["PATH"] = os.path.join(root, "bin") + os.pathsep + os.environ.get("PATH", "")
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)

    port = free_port()
    threading.Thread(target=serve, args=(port,), daemon=True).start()
    write_port(port)
    if not wait_ready(port):
        # 起不来就说清楚，不要开一个白窗口让人猜
        webview.create_window(APP_NAME, html=(
            "<body style='font:15px -apple-system;padding:40px;color:#333'>"
            "<h3>Pipo 没能启动</h3><p>本机服务未在 30 秒内就绪。</p></body>"))
        webview.start()
        return 1

    api = Bridge()
    win = webview.create_window(
        APP_NAME, "http://127.0.0.1:%d/" % port,
        width=1280, height=860, min_size=(960, 640), js_api=api)
    api.window = win
    try:
        webview.start(wire_drop, win)
    finally:
        # 窗口一关进程就退，端口文件必须跟着删 —— 留着会让技能
        # 连一个已经不存在的端口，报的是「连不上」而不是「没启动」
        clear_port()
    return 0


def wire_drop(window) -> None:
    """把拖放接到原生侧。

    **拖放必须在 Python 这边处理**，页面里做不到：浏览器的 drop 事件给的是
    File 对象，只有内容没有路径，而本地版整个不拷贝的设计要的就是路径。
    pywebview 的 cocoa 后端会在原生层把拖进来的文件路径塞进事件里
    （pywebviewFullPath），但只有在 Python 侧注册过 drop 监听时才收集 ——
    所以这个函数不能省。
    """
    def on_drop(e):
        paths = [f["pywebviewFullPath"] for f in e.get("dataTransfer", {}).get("files", [])
                 if f.get("pywebviewFullPath")]
        if paths:
            window.evaluate_js("window.addLocal(%s)" % json.dumps(paths))

    try:
        window.dom.get_element("body").events.drop += on_drop
    except Exception:
        # 拖放接不上不该让 app 起不来 —— 还有「导入素材」按钮那条路
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())

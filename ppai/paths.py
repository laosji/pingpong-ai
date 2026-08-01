"""数据目录的唯一定义处。

**为什么要有这个模块。** 打包成 .app 之后代码住在应用包里，那是只读的，
而 `cache/`、`out/`、`pipo.db` 都要写。原来这些路径散落在三处：
server.py 用 ROOT 拼、store.py 用相对的 "pipo.db"、embed/rerank 用相对的
"cache" —— 后两个相对的是**当前工作目录**，而 app 启动时 CWD 是什么完全
不由我们决定（Finder 双击时通常是 /）。三处各写各的，迟早有一处漏掉，
表现是「开发机好好的，装到别人电脑上缓存不命中」这种很难查的问题。

PIPO_DATA_DIR 不设时退回仓库根目录，开发时的行为和以前完全一样。
"""
from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.abspath(os.environ.get("PIPO_DATA_DIR", REPO))


def sub(name: str, make: bool = False) -> str:
    p = os.path.join(DATA, name)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


CACHE = sub("cache")
OUT = sub("out")
LABELS = sub("labels")
UPLOADS = sub("uploads")
DB = os.environ.get("PIPO_DB") or os.path.join(DATA, "pipo.db")

# 模型是**只读资源**，跟着代码走，不进数据目录 —— 打包时它们在应用包里。
MODELS = os.path.join(REPO, "models")

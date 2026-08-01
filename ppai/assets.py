"""按需下载的大文件。

为什么不打进安装包
------------------
cnn14.onnx 是 323MB，占安装包的七成（打包实测 .app 683MB / .dmg 454MB，
去掉它是 ~130MB）。它是**冻结的**预训练骨干，我们不训它、也不会更新，
所以没有理由让每次发版都重新分发一遍 323MB。

**但它不是可选的。** 缺了它：
  * `rerank.apply` 原样返回候选 —— 击球检测准确率从 0.68 掉回 0.38
  * `looks_like_pingpong` 按设计返回 ok=True —— 乒乓球门禁静默放行一切

两者都不报错，只让成片悄悄变差。所以这里的策略是**挡住而不是降级**：
没有模型就不让开始分析，明确提示去下载，而不是"先凑合跑"。

查找顺序：数据目录 → 应用包内。前者是下载落点，后者留给开发环境
（仓库里就有一份）和将来可能的完整包。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

from .paths import DATA, MODELS


# 发版时内置的下载地址。环境变量可以覆盖（开发和自建镜像用）。
# R2 的 r2.dev 地址**有速率限制、Cloudflare 明说不适合生产**，
# 正式发版前应该换成绑定的自有域名 —— 换域名只改这一行。
DEFAULT_BASE = "https://pub-a7dae8fcbe6a4b418f443b1528d3114e.r2.dev/v1"


class Asset:
    def __init__(self, name: str, filename: str, size: int, sha256: str, note: str,
                 remote: str = "", dl_size: int = 0, dl_sha256: str = ""):
        self.name = name
        self.filename = filename          # 落到磁盘上的最终文件名
        self.size = size                  # 解压后的体积
        self.sha256 = sha256              # **解压后**的校验和 —— 这才是要用的那个
        self.note = note
        self.remote = remote or filename  # 远端文件名（可能带 .gz）
        self.dl_size = dl_size or size
        self.dl_sha256 = dl_sha256        # 下载物本身的校验和，空则跳过这一层

    @property
    def gzipped(self) -> bool:
        return self.remote.endswith(".gz")

    @property
    def url(self) -> str:
        """下载地址。环境变量优先，其次是发版内置的。"""
        base = (os.environ.get("PIPO_ASSET_BASE") or DEFAULT_BASE).rstrip("/")
        return "%s/%s" % (base, self.remote) if base else ""

    def path(self) -> Optional[str]:
        """已经就位的路径，没有返回 None。"""
        for p in (os.path.join(DATA, "models", self.filename),
                  os.path.join(MODELS, self.filename)):
            if os.path.exists(p) and os.path.getsize(p) > self.size * 0.95:
                return p
        return None

    def target(self) -> str:
        d = os.path.join(DATA, "models")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, self.filename)


ASSETS: List[Asset] = [
    Asset(
        name="detector",
        filename="cnn14.onnx",
        size=323011186,
        sha256="22b41aef1908df6612fb9cf18640519d5b1398b7c79d3d46b0f5856470b1e3e0",
        note="声学模型，用来分辨真正的击球声和球台碰撞、脚步、说话",
        # 传 gzip 版有两个原因：一是给用户省 20MB 下载，
        # 二是 wrangler 的上传上限是 300 MiB 而原始文件 308 MiB，
        # 压完 287.7 MiB 刚好进得去。float32 权重只能压到 93%，
        # **余量很薄** —— 模型再大一点就得改用 S3 分片上传。
        remote="cnn14.onnx.gz",
        dl_size=301692277,
        dl_sha256="42c243ca2ea7055c4df5e0311ebc692626c653741cc7db30957c5a753b856279",
    ),
]


def missing() -> List[Asset]:
    return [a for a in ASSETS if a.path() is None]


def status() -> Dict:
    out = []
    for a in ASSETS:
        p = a.path()
        out.append({
            "name": a.name, "filename": a.filename, "note": a.note,
            "size": a.size, "ready": p is not None,
            "mb": round(a.size / 1e6),
            "configured": bool(a.url),
        })
    return {"assets": out, "ready": not missing()}


def sha256_of(path: str, on_progress: Optional[Callable[[float], None]] = None) -> str:
    h = hashlib.sha256()
    total = os.path.getsize(path)
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
            done += len(b)
            if on_progress and total:
                on_progress(done / total)
    return h.hexdigest()


def download(a: Asset, on_progress: Optional[Callable[[int, int], None]] = None,
             timeout: float = 60.0) -> str:
    """下载并校验，成功后原子替换到目标路径。

    **先下到临时文件再改名**：中途断网/退出会留下一个半截文件，
    如果直接写目标路径，下次启动会看到一个「存在但不完整」的模型 ——
    ONNX 加载时才报错，而且报的是格式错误，很难联想到是没下完。

    校验 sha256 不是防篡改（那要签名），是防**下错和下不全**。
    CDN 返回一个 HTML 错误页、代理截断、磁盘满，都会得到一个能存下来
    但用不了的文件。
    """
    if not a.url:
        raise RuntimeError("没有配置下载地址（PIPO_ASSET_BASE）")

    tmp_dir = os.path.join(DATA, "models")
    os.makedirs(tmp_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=tmp_dir, prefix=".dl_", suffix=".part")
    os.close(fd)
    try:
        req = urllib.request.Request(a.url, headers={"User-Agent": "Pipo"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            total = int(r.headers.get("Content-Length") or a.dl_size)
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(done, total)
        # 两层校验。下载物这一层挡「没下全 / CDN 返回错误页」，
        # 解压后这一层挡「解压出来是坏的」—— 少任何一层都会让一个
        # 能存下来但用不了的模型混过去，而 ONNX 要到加载时才报错。
        if a.dl_sha256:
            got = sha256_of(tmp)
            if got != a.dl_sha256:
                raise RuntimeError(
                    "下载校验不通过：期望 %s…，实得 %s…（多半是没下全）"
                    % (a.dl_sha256[:12], got[:12]))

        if a.gzipped:
            import gzip
            fd2, raw = tempfile.mkstemp(dir=tmp_dir, prefix=".un_", suffix=".part")
            os.close(fd2)
            try:
                with gzip.open(tmp, "rb") as src, open(raw, "wb") as dst_f:
                    shutil.copyfileobj(src, dst_f, 1 << 22)
            except Exception:
                os.remove(raw)
                raise
            os.remove(tmp)
            tmp = raw

        got = sha256_of(tmp)
        if got != a.sha256:
            raise RuntimeError(
                "校验不通过：期望 %s…，实得 %s…（多半是没下全或下到了错的东西）"
                % (a.sha256[:12], got[:12]))
        dst = a.target()
        os.replace(tmp, dst)          # 同目录内，原子
        return dst
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def cleanup_partials() -> int:
    """清掉上次中断留下的半截文件。启动时跑一次。"""
    d = os.path.join(DATA, "models")
    n = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.startswith((".dl_", ".un_")) and f.endswith(".part"):
                try:
                    os.remove(os.path.join(d, f))
                    n += 1
                except OSError:
                    pass
    return n

"""用户反馈回流 —— 把本机的负例标注打成一个匿名包。

为什么值得做
------------
检测器的召回已经 0.850，问题全在准确率（0.38-0.49）—— 也就是**误报**。
误报来自球台碰撞、脚步、说话、场馆混响，而这些恰恰是每个球馆各不相同、
我们自己攒不出来的东西。用户点「这段没在打球」产生的正是这类负例，
是唯一能靠人数增长而不是靠我们标注增长的数据。

只回流「明确说了没在打球」的区间
--------------------------------
**删除片段本身不算负例。** 用户删一段的理由可能只是想剪短，那段其实
打得好好的 —— 把真实回合当负例喂回去会反向污染模型，而且是所有用户
一起污染。所以前端删完会再问一句，只有明确回答「检测错了」才写进
negative_ranges，也只有这些才进这个包。宁可少收，不收含义不明的。

包里有什么、没有什么
--------------------
有：候选时刻（相对区间起点的秒数）+ 每个候选的 2048 维嵌入（float16）
没有：音频、画面、文件名、文件路径、拍摄时间、机器名

嵌入是 CNN14 对 1 秒窗口的 2048 维输出。它不是音频，也没有已知的方法
从中还原出可听的声音 —— 但我要说清楚这是「据我所知」，不是数学保证，
嵌入反演在别的模型上是有过成功案例的。所以这件事必须用户明确同意才做，
不能默认打开。

为什么带一个随机安装 id
-----------------------
只为去重：同一台机器反复发同一段反馈时能认出来。它是本地生成的随机数，
不含任何设备信息，删掉 settings.json 就换一个，我们无法用它反查到人。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import time
from typing import Dict, List, Optional

import numpy as np

SCHEMA = 1


def _settings_path(data_dir: str) -> str:
    return os.path.join(data_dir, "settings.json")


def settings(data_dir: str) -> Dict:
    p = _settings_path(data_dir)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            s = json.load(fh)
    except Exception:
        s = {}
    # 安装 id 在第一次读设置时生成，但**这不等于开启回流** ——
    # 没有它就没法在用户同意后去重，有它也不会自己发送任何东西。
    if not s.get("install_id"):
        s["install_id"] = secrets.token_hex(8)
        save_settings(data_dir, s)
    s.setdefault("contribute", False)     # 默认关闭，必须用户明确打开
    s.setdefault("sent", [])              # 已发过的条目 id，避免重复上传
    return s


def save_settings(data_dir: str, s: Dict) -> None:
    p = _settings_path(data_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, p)                    # 原子替换，别把设置写坏


def _item_id(install_id: str, video_fp: str, a: float, b: float) -> str:
    """条目 id：安装 id + 视频指纹 + 区间。

    加了 install_id 当盐，所以**不同用户的同一段视频算出来的 id 不同** ——
    否则这个 id 就成了「这两个人有同一个文件」的探针。
    """
    h = hashlib.blake2b(digest_size=12)
    h.update(("%s|%s|%.3f|%.3f" % (install_id, video_fp, a, b)).encode())
    return h.hexdigest()


def collect(label_dir: str, cfg: Dict, data_dir: str,
            fp_of=None, only_new: bool = True) -> Dict:
    """扫本机标注，收集「用户明确说没在打球」的区间。

    返回 {"items":[...], "n_ranges":int, "n_cand":int, "bytes":int}
    items 里每条是 {id, sec, times, feats}，**不含任何路径或文件名**。

    fp_of(video_path) -> 指纹字符串；拿不到就退回路径的哈希（只用来算
    条目 id，且已被 install_id 加盐，不会离开本机）。
    """
    from . import labels as L, rerank

    s = settings(data_dir)
    sent = set(s.get("sent") or [])
    items: List[Dict] = []
    n_ranges = n_cand = 0

    import glob
    for path in sorted(glob.glob(os.path.join(label_dir, "*.json"))):
        try:
            lab = L.load(path)
        except Exception:
            continue
        neg = lab.get("negative_ranges") or []
        if not neg:
            continue
        frz = rerank._frozen(path)
        if frz is None:
            # 没固化就没得发 —— 原片可能已经不在了，这里不去重算
            continue
        det, feats = frz
        video = lab.get("video") or path
        fp = fp_of(video) if fp_of else hashlib.blake2b(
            video.encode(), digest_size=12).hexdigest()
        for a, b in neg:
            iid = _item_id(s["install_id"], fp, float(a), float(b))
            if only_new and iid in sent:
                continue
            m = (det >= a) & (det <= b)
            if not m.any():
                continue
            n_ranges += 1
            n_cand += int(m.sum())
            items.append({
                "id": iid,
                "sec": round(float(b) - float(a), 3),
                # 时刻转成**相对区间起点**的秒数：绝对时刻会泄露
                # 「这段录像有多长、事件在第几分钟」这类无关信息
                "times": np.round(det[m] - float(a), 3).astype(np.float32),
                "feats": feats[m].astype(np.float16),
            })

    return {"items": items, "n_ranges": n_ranges, "n_cand": n_cand,
            "bytes": sum(x["times"].nbytes + x["feats"].nbytes for x in items)}


def pack(bundle: Dict, install_id: str, app_version: str = "") -> bytes:
    """打成一个 npz。**不带 pickle**（allow_pickle 关掉），只有数组和一段 JSON。"""
    items = bundle["items"]
    meta = {
        "schema": SCHEMA,
        "install": install_id,
        "app": app_version,
        "created": int(time.time()),
        "items": [{"id": x["id"], "sec": x["sec"], "n": int(len(x["times"]))}
                  for x in items],
    }
    arrays = {"meta": np.frombuffer(
        json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)}
    for i, x in enumerate(items):
        arrays["t%d" % i] = x["times"]
        arrays["f%d" % i] = x["feats"]
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def mark_sent(data_dir: str, ids: List[str]) -> None:
    s = settings(data_dir)
    s["sent"] = sorted(set(s.get("sent") or []) | set(ids))
    s["last_sent"] = int(time.time())
    save_settings(data_dir, s)


def upload(blob: bytes, url: str, install_id: str, timeout: float = 30.0) -> Dict:
    """发到收集端。只发一次，不重试 —— 失败了下次再发就是了，
    这是可以慢慢来的数据，不值得为它写重试队列。"""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url, data=blob, method="POST",
        headers={"Content-Type": "application/octet-stream",
                 "X-Pipo-Install": install_id,
                 "X-Pipo-Schema": str(SCHEMA)})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"ok": 200 <= r.status < 300, "status": r.status}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "error": e.reason}
    except Exception as e:
        return {"ok": False, "status": 0, "error": str(e)}

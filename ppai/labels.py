"""标注格式与读写。

设计取舍：主标注是**区间级**（打球/没打球），不是逐拍级。
理由有三：
  1. 区间级就是模块四要的产品指标，也是方案「第一阶段训练目标」的形式；
  2. 一个视频几分钟标完，逐拍标要点几千次；
  3. 区间级足够训练分类器 —— 把区间切成 1 秒窗即为样本。
逐拍标注（hits）是可选的，留给模块五算回合拍数时再补。

阴性素材（没有乒乓球的视频）是免费真值：整段 not_playing，零人工。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

SCHEMA = 1


def empty(video: str, duration: float, complete: bool = False) -> Dict:
    return {
        "schema": SCHEMA,
        "video": os.path.abspath(video),
        "duration": round(duration, 3),
        # complete=True 表示整段都看过了。整片穷尽标注很费时，实际很少这么做。
        "complete": complete,
        # 更实用的形式：只声明**某几段**是穷尽标注的。
        # 准确率只能在这些区间内计算 —— 区间外没有标记不代表那里没发生，
        # 把它当负样本会把漏标算成误报，得出的准确率毫无意义。
        # 召回则不受影响，全部标记都可用。
        "complete_ranges": [],
        # 用户在产品里主动否定的区间（「这段不对」）。与 complete_ranges 分开：
        # 后者零击球会被判为标注遗漏并跳过，而这里的零击球是**用户明确声称**的，
        # 是可信的纯负例。混在一起会让保护机制把真实反馈也挡掉。
        "negative_ranges": [],
        "playing": [],      # [[start, end], ...] 击球/有效比赛
        # 捡球区间：球落地后到重新开始之间。和 playing 分开存 ——
        # 这是**负例**，混进 playing 会让真值直接失效。
        "pickup": [],
        "hits": [],         # 可选：击球瞬态时间点
        # 发球时间点。回合边界现在靠「静音间隔 > gap_s」猜，实测很脆：
        # 34 个长回合里 27 个的内部最大间隔都在阈值 70% 以上，
        # 阈值从 0.65 降到 0.60，最长回合就从 39 拍裂成 29 拍。
        # 发球是回合起点的真值，标它比穷尽标击球便宜约 10 倍
        # （11 次/分钟 vs 100 次/分钟）。
        "serves": [],
        "notes": "",
    }


def negative(video: str, duration: float, note: str = "阴性对照：无乒乓球") -> Dict:
    """整段判定为没打球。用于有声书、影视剧等对照素材。"""
    lab = empty(video, duration, complete=True)
    lab["notes"] = note
    return lab


def path_for(video: str, label_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(label_dir, stem + ".json")


VIDEO_DIRS_ENV = "PIPO_VIDEO_DIR"


def resolve_video(ref: str) -> str:
    """把标注里的素材引用解析成本机的绝对路径。

    **标注文件里存的是文件名，不是绝对路径。** 存绝对路径有两个问题：
    换台机器就找不到，而且路径本身会带上用户名和目录结构 ——
    标注是要进公开仓库的训练数据，不该夹带这些。

    解析顺序：本来就是能用的绝对路径 → PIPO_VIDEO_DIR（可用冒号分隔多个）
    → 常见位置。都找不到就原样返回，调用方自己判断存不存在
    （build_dataset 有固化嵌入的兜底，原片不在也能训练）。
    """
    if os.path.isabs(ref) and os.path.exists(ref):
        return ref
    name = os.path.basename(ref)
    roots = []
    env = os.environ.get(VIDEO_DIRS_ENV, "")
    roots += [p for p in env.split(os.pathsep) if p]
    roots += [os.path.join(os.path.expanduser("~"), "Downloads", "PP-video"),
              os.path.join(os.getcwd(), "videos"),
              os.path.join(os.getcwd(), "negatives"),   # 阴性对照放这儿
              os.getcwd()]
    for r in roots:
        p = os.path.join(os.path.expanduser(r), name)
        if os.path.exists(p):
            return p
    return ref


def load(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        lab = json.load(f)
    # 存的是文件名，用的是绝对路径 —— 在这里解析，
    # 所有读 lab["video"] 的地方就都不用改
    if lab.get("video"):
        lab["video"] = resolve_video(lab["video"])
    if lab.get("schema") != SCHEMA:
        raise ValueError("标注格式版本不符: %s" % path)
    lab["playing"] = _normalize(lab.get("playing", []), lab["duration"])
    lab["pickup"] = _normalize(lab.get("pickup", []), lab["duration"])
    lab["negative_ranges"] = _normalize(lab.get("negative_ranges", []), lab["duration"])
    lab["complete_ranges"] = _normalize(lab.get("complete_ranges", []), lab["duration"])
    if lab.get("complete") and not lab["complete_ranges"]:
        lab["complete_ranges"] = [[0.0, lab["duration"]]]
    return lab


def scored_ranges(lab: Dict) -> List[List[float]]:
    """可用于算准确率的时段。空列表表示这份标注只能算召回。"""
    return lab.get("complete_ranges") or []


def save(lab: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lab["playing"] = _normalize(lab.get("playing", []), lab["duration"])
    lab["pickup"] = _normalize(lab.get("pickup", []), lab["duration"])
    lab["negative_ranges"] = _normalize(lab.get("negative_ranges", []), lab["duration"])
    out = dict(lab)
    # 落盘时只留文件名。绝对路径换台机器就失效，而且会把用户名和
    # 目录结构写进要进公开仓库的训练数据里。
    if out.get("video"):
        out["video"] = os.path.basename(out["video"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path


def _normalize(intervals: List, duration: float) -> List[List[float]]:
    """排序、裁剪到时长内、合并重叠。手工拖出来的区间常常有重叠。"""
    out: List[List[float]] = []
    for iv in sorted([[max(0.0, float(a)), min(duration, float(b))] for a, b in intervals]):
        if iv[1] - iv[0] <= 1e-6:
            continue
        if out and iv[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], iv[1])
        else:
            out.append(iv)
    return [[round(a, 3), round(b, 3)] for a, b in out]


def to_mask(intervals: List, duration: float, dt: float) -> np.ndarray:
    """区间 -> 时间网格上的布尔掩码，评测用。"""
    n = max(1, int(np.ceil(duration / dt)))
    mask = np.zeros(n, dtype=bool)
    for a, b in intervals:
        mask[int(a / dt):int(np.ceil(b / dt))] = True
    return mask


def find(video: str, label_dir: str) -> Optional[Dict]:
    p = path_for(video, label_dir)
    return load(p) if os.path.exists(p) else None


def audit(lab: Dict) -> List[Dict]:
    """检查一份标注有没有会**静默污染训练集**的问题。返回问题列表。

    为什么需要这个
    --------------
    build_dataset 里原有的守卫只在「某个 complete_range 内**零** playing、
    却有 pickup」时才触发 —— 它挡不住「标到一半换错轨道」。
    实测碰到过一份：前 19 秒 playing 正常，之后 99 段击球全标进了 pickup，
    守卫因为看到前面有 playing 就放行了。结果是声明穷尽的 120 秒里
    514 个候选只有 90 个算真击球（18%，正常是 38-51%），
    **424 个真击球被当成负例喂进模型**。比不标这份还糟。

    三条判据，都是这次实际用来定位问题的：

      1. **形状**：击球是零点一秒的瞬态，捡球是持续几秒的一段。
         pickup 里出现大量与 playing 同样时长的短区间 = 标错轨道了。
      2. **发球位置**：捡球的时候不会发球。发球落在 pickup 区间里 = 那段在打球。
      3. **声明与实际不符**：complete_ranges 说标完了，但后面一段根本没有任何标注。
    """
    out: List[Dict] = []
    pl = [x for x in (lab.get("playing") or []) if len(x) == 2]
    pk = [x for x in (lab.get("pickup") or []) if len(x) == 2]
    sv = list(lab.get("serves") or [])
    rng = scored_ranges(lab)

    # 1) pickup 里混进了击球形状的短区间
    if pk and pl:
        dp = sorted(b - a for a, b in pl)
        dk = [b - a for a, b in pk]
        med_pl = dp[len(dp) // 2]
        short = [d for d in dk if d <= max(0.25, med_pl * 2.5)]
        if len(short) >= 10 and len(short) / len(dk) >= 0.5:
            out.append({
                "level": "block",
                "what": "pickup 轨道里有 %d/%d 段是击球形状（中位 %.3f 秒，"
                        "playing 中位 %.3f 秒）—— 多半是标注时切错了轨道"
                        % (len(short), len(dk), sorted(dk)[len(dk) // 2], med_pl),
            })

    # 2) 发球落在**真捡球区间内部** —— 捡球时不发球。
    #
    # 第一版用 ±1.5 秒的容差，结果在一份**正确**的标注上误报了 24/45：
    # 实际节奏是「捡完球立刻发」，发球在捡球结束后中位 0.13 秒。
    # 紧邻是对的，落在里面才是错的。
    #
    # 而且只看**长于 0.5 秒**的 pickup —— 短的那些本身就可能是标错轨道的击球，
    # 拿它们当参照会互相印证出一个假结论。
    if sv and pk:
        real_pk = [(a, b) for a, b in pk if b - a > 0.5]
        inside = sum(1 for s in sv if any(a <= s <= b for a, b in real_pk))
        if real_pk and inside >= max(2, len(sv) * 0.5):
            out.append({
                "level": "block",
                "what": "%d/%d 个发球落在 pickup 区域里 —— 捡球时不会发球，"
                        "那一段应该是在打球" % (inside, len(sv)),
            })

    # 3) 声明穷尽的区间尾部没有任何标注。
    #
    # **必须逐区间算，不能用全局最后一个标注。** 第一版用 max(所有标注)，
    # 结果一份有两个区间的标注（[60,180] 标到 146、[220,269] 标满）漏报了：
    # 全局 last=269 比 180 大，第一个区间的检查就永远不会触发。
    # 校验器在它该抓的场景上静默放行，比没有校验器更糟。
    marks = [b for _, b in pl] + [b for _, b in pk] + list(sv)
    if rng and marks:
        for a, b in rng:
            inside = [t for t in marks if a - 0.5 <= t <= b + 0.5]
            if not inside:
                out.append({
                    "level": "block",
                    "what": "complete_ranges 声明标完了 %.0f-%.0f 秒，"
                            "但这个区间里一处标注都没有" % (a, b),
                })
                continue
            last = max(inside)
            if b - last > 5.0:
                out.append({
                    "level": "block",
                    "what": "complete_ranges 声明标完了 %.0f-%.0f 秒，"
                            "但这个区间里最后一处标注在 %.1f 秒 —— "
                            "空着的 %.0f 秒里的真击球会被当成负例"
                            % (a, b, last, b - last),
                })

    # 4) 真击球占比异常低（正常 0.38-0.51）——「可能哪里标漏了」的兜底信号
    if rng and pl:
        span = sum(b - a for a, b in rng)
        if span > 30 and len(pl) / span < 0.25:
            out.append({
                "level": "warn",
                "what": "声明穷尽的 %.0f 秒里只标了 %d 次击球（%.2f 次/秒），"
                        "偏稀疏，确认一下是不是漏了" % (span, len(pl), len(pl) / span),
            })
    return out

"""用户管理（内测用）。

  python -m ppai.users add 张三          新建用户，打印魔法链接
  python -m ppai.users list              列出所有用户
  python -m ppai.users grant <uid> <视频路径>...   把已有文件登记到某用户名下
  python -m ppai.users rm <uid>          删除用户
  python -m ppai.users invite [个数] [每码可剪几个视频] [有效天数]
                                        生成邀请码，默认 20 个 / 3 个视频 / 60 天
  python -m ppai.users invites           列出邀请码和认领情况
"""
from __future__ import annotations

import os
import sys

from . import store
from .media import probe

BASE = os.environ.get("PIPO_BASE_URL", "http://127.0.0.1:8020")


def _link(tok: str) -> str:
    return "%s/enter?t=%s" % (BASE.rstrip("/"), tok)


def main(argv=None) -> int:
    a = (argv or sys.argv[1:])
    if not a:
        print(__doc__); return 1
    cmd = a[0]

    if cmd == "add" and len(a) >= 2:
        u = store.create_user(" ".join(a[1:]))
        print("已创建: %s (%s)" % (u["name"], u["id"]))
        print("魔法链接（发给本人，点一次即可，之后靠 cookie）：")
        print("  " + _link(u["token"]))
        return 0

    if cmd == "invite":
        n = int(a[1]) if len(a) > 1 else 20
        q = int(a[2]) if len(a) > 2 else 3
        d = float(a[3]) if len(a) > 3 else 60
        codes = store.make_invites(n, quota=q, days=d)
        print("已生成 %d 个邀请码，每个可剪 %d 个视频，%.0f 天有效：" % (n, q, d))
        for c in codes:
            print("  " + c)
        return 0

    if cmd == "invites":
        iv = store.list_invites()
        if not iv:
            print("还没有邀请码。用 `invite` 生成。"); return 0
        names = {u["id"]: u["name"] for u in store.list_users()}
        free = sum(1 for x in iv if not x["claimed_by"])
        print("共 %d 个，未认领 %d 个\n" % (len(iv), free))
        print("%-14s %5s %s" % ("邀请码", "配额", "认领者"))
        for x in iv:
            who = names.get(x["claimed_by"], x["claimed_by"] or "—")
            print("%-14s %5d %s" % (x["code"], x["quota"], who))
        return 0

    if cmd == "list":
        us = store.list_users()
        if not us:
            print("还没有用户。用 `add <名字>` 创建。"); return 0
        print("%-12s %-14s %6s %s" % ("id", "名字", "视频数", "魔法链接"))
        for u in us:
            n = len(store.list_videos(u["id"]))
            print("%-12s %-14s %6d %s" % (u["id"], u["name"][:14], n, _link(u["token"])))
        return 0

    if cmd == "rm" and len(a) >= 2:
        print("已删除 %d 个用户" % store.delete_user(a[1])); return 0

    if cmd == "grant" and len(a) >= 3:
        uid = a[1]
        if not any(x["id"] == uid for x in store.list_users()):
            print("没有这个用户: %s" % uid); return 1
        import server                    # 复用同一套指纹算法
        n = 0
        for p in a[2:]:
            p = os.path.abspath(p)
            if not os.path.exists(p):
                print("  跳过（不存在）: %s" % p); continue
            try:
                m = probe(p)
            except Exception as e:
                print("  跳过（无法解析）: %s" % os.path.basename(p)); continue
            store.add_video(uid, {
                "path": p, "name": os.path.basename(p), "fp": server._fingerprint(p),
                "duration": m["duration"], "width": m["width"], "height": m["height"],
                "quality": m["quality_score"], "note": m["recommendation"],
                "note_en": m.get("recommendation_en", "")})
            print("  + %s" % os.path.basename(p)); n += 1
        print("登记 %d 个视频到 %s" % (n, uid))
        return 0

    print(__doc__); return 1


if __name__ == "__main__":
    sys.exit(main())

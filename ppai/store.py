"""持久化 + 用户归属（SQLite）。

为什么是 SQLite：内测规模下不需要独立数据库进程，单文件、事务安全、
标准库自带。真到需要横向扩展时再换，接口不用动。

三张表就够：
  users   —— 谁
  videos  —— 谁上传的哪个视频（隔离的依据）
  jobs    —— 任务状态，重启不丢

鉴权用「一人一条魔法链接」而不是账号密码：内测发给十几个熟人，
密码意味着要做注册、找回、加密存储一整套，而收益只是多一层用户自选的弱口令。
token 由服务端生成（32 字节随机），用户点一次链接就种 cookie，之后不用再管。
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import time
from typing import Dict, List, Optional

from .paths import DB as DB_PATH
_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, token TEXT NOT NULL UNIQUE,
  is_admin INTEGER DEFAULT 0, created REAL NOT NULL, last_seen REAL);
CREATE TABLE IF NOT EXISTS videos(
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, path TEXT NOT NULL,
  name TEXT NOT NULL, fp TEXT NOT NULL, duration REAL, width INTEGER,
  height INTEGER, quality INTEGER, note TEXT, note_en TEXT, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, video_id TEXT,
  state TEXT NOT NULL, stage TEXT, payload TEXT, created REAL NOT NULL, updated REAL);
CREATE TABLE IF NOT EXISTS invites(
  code TEXT PRIMARY KEY, quota INTEGER NOT NULL, note TEXT,
  created REAL NOT NULL, expires REAL,
  claimed_by TEXT, claimed_at REAL, email TEXT, emailed_at REAL);
CREATE TABLE IF NOT EXISTS oauth_codes(
  code TEXT PRIMARY KEY, user_id TEXT NOT NULL, client_id TEXT,
  redirect_uri TEXT, challenge TEXT, created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_videos_user ON videos(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_user_fp ON videos(user_id, fp);
"""


def conn() -> sqlite3.Connection:
    """每线程一个连接 —— sqlite3 的连接不能跨线程用，而任务跑在后台线程里。"""
    c = getattr(_local, "conn", None)
    if c is None:
        c = sqlite3.connect(DB_PATH, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")   # 读写并发：任务线程写时页面仍可读
        c.execute("PRAGMA foreign_keys=ON")
        c.executescript(SCHEMA)
        # SCHEMA 里是 CREATE TABLE IF NOT EXISTS，加字段不会作用到已有的库，
        # 所以新列要单独 ALTER。重复执行会报 duplicate column，忽略即可。
        for tbl, col, decl in (("videos", "shot_at", "REAL"),
                               ("users", "invite", "TEXT"),
                               ("users", "quota", "INTEGER"),
                               ("invites", "email", "TEXT"),
                               ("invites", "emailed_at", "REAL")):
            try:
                c.execute("ALTER TABLE %s ADD COLUMN %s %s" % (tbl, col, decl))
                c.commit()
            except sqlite3.OperationalError:
                pass
        _local.conn = c
    return c


def _row(r) -> Optional[Dict]:
    return dict(r) if r is not None else None


# ── 用户 ──────────────────────────────────────────────────
def create_user(name: str, is_admin: bool = False,
                invite: Optional[str] = None,
                quota: Optional[int] = None) -> Dict:
    """quota=None 表示不限量（管理员和早期用户）；有值时是「能剪几个视频」。"""
    uid, tok = secrets.token_hex(8), secrets.token_urlsafe(24)
    c = conn()
    c.execute("""INSERT INTO users(id,name,token,is_admin,created,invite,quota)
                 VALUES(?,?,?,?,?,?,?)""",
              (uid, name, tok, int(is_admin), time.time(), invite, quota))
    c.commit()
    return {"id": uid, "name": name, "token": tok, "is_admin": is_admin,
            "invite": invite, "quota": quota}


def local_user(uid: str = "local") -> Dict:
    """桌面 app 的唯一用户，不存在就建。

    仍然走 users 表而不是绕过它：视频归属、任务、反馈全部外键到 user_id，
    为了单机模式给它们各开一条无主分支，只会让两种模式慢慢跑偏。
    一行记录换来所有下游代码不用改。

    quota 恒为 None（不限量）—— 配额是配给服务器资源的，本地没有。
    """
    c = conn()
    r = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if r:
        return dict(r)
    c.execute("""INSERT INTO users(id,name,token,is_admin,created,invite,quota)
                 VALUES(?,?,?,?,?,?,?)""",
              (uid, "本机", secrets.token_urlsafe(24), 1, time.time(), None, None))
    c.commit()
    return dict(c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())


# ── 邀请码 ────────────────────────────────────────────────
def make_invites(n: int, quota: int = 3, days: Optional[float] = None,
                 note: str = "") -> List[str]:
    """生成一批邀请码。quota 是「这个码开出来的账号能剪几个视频」。

    码用 12 个 base32 字符（去掉易混的 0/1/O/I），大约 60 bit ——
    公开发放时不能用短码，否则能被枚举。
    """
    ab = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    out, c = [], conn()
    exp = time.time() + days * 86400 if days else None
    for _ in range(n):
        code = "".join(secrets.choice(ab) for _ in range(12))
        c.execute("""INSERT INTO invites(code,quota,note,created,expires)
                     VALUES(?,?,?,?,?)""", (code, quota, note, time.time(), exp))
        out.append(code)
    c.commit()
    return out


def set_invite_email(code: str, email: str) -> None:
    c = conn()
    c.execute("UPDATE invites SET email=?, emailed_at=? WHERE code=?",
              (email, time.time(), code))
    c.commit()


def invite_by_email(email: str) -> Optional[Dict]:
    """同一个邮箱已经领过就把原来那个还给他，而不是再发一个新的。"""
    return _row(conn().execute(
        "SELECT * FROM invites WHERE email=? ORDER BY emailed_at DESC LIMIT 1",
        (email,)).fetchone())


def take_invite(code: str, user_id: str) -> Optional[Dict]:
    """认领一个邀请码。**一码一人** —— 认领后就绑死，别人再输同一个码无效。

    用 UPDATE ... WHERE claimed_by IS NULL 而不是「先查再写」：
    两个人同时提交同一个码时，先查再写会让两人都通过。
    """
    c = conn()
    now = time.time()
    n = c.execute("""UPDATE invites SET claimed_by=?, claimed_at=?
                     WHERE code=? AND claimed_by IS NULL
                       AND (expires IS NULL OR expires > ?)""",
                  (user_id, now, code, now)).rowcount
    c.commit()
    if not n:
        return None
    return _row(c.execute("SELECT * FROM invites WHERE code=?", (code,)).fetchone())


def free_invite() -> Optional[Dict]:
    """取一个还没被认领的码，给「点一下领取」用。"""
    now = time.time()
    return _row(conn().execute(
        """SELECT * FROM invites WHERE claimed_by IS NULL
           AND (expires IS NULL OR expires > ?) ORDER BY created LIMIT 1""",
        (now,)).fetchone())


def list_invites() -> List[Dict]:
    return [dict(r) for r in
            conn().execute("SELECT * FROM invites ORDER BY created DESC").fetchall()]


def used_quota(user_id: str) -> int:
    """已用配额 = **出过片的不同视频数**，不是任务数。

    同一个视频换主题重剪不扣次数 —— 否则用户会不敢试主题，
    而试主题恰恰是这产品的核心动作。
    """
    r = conn().execute(
        """SELECT COUNT(DISTINCT video_id) n FROM jobs
           WHERE user_id=? AND state='done' AND video_id IS NOT NULL""",
        (user_id,)).fetchone()
    return int(r["n"] or 0)


def set_user_quota(uid: str, invite: str, quota: Optional[int]) -> None:
    c = conn()
    c.execute("UPDATE users SET invite=?, quota=? WHERE id=?", (invite, quota, uid))
    c.commit()


def done_video_ids(user_id: str) -> set:
    """已经成功出过片的视频 id。配额按这个算，不按任务数。"""
    return {r["video_id"] for r in conn().execute(
        """SELECT DISTINCT video_id FROM jobs
           WHERE user_id=? AND state='done' AND video_id IS NOT NULL""",
        (user_id,)).fetchall()}


def user_by_token(token: str) -> Optional[Dict]:
    if not token:
        return None
    c = conn()
    r = c.execute("SELECT * FROM users WHERE token=?", (token,)).fetchone()
    if r:
        c.execute("UPDATE users SET last_seen=? WHERE id=?", (time.time(), r["id"]))
        c.commit()
    return _row(r)


def list_users() -> List[Dict]:
    return [dict(r) for r in
            conn().execute("SELECT * FROM users ORDER BY created").fetchall()]


def delete_user(uid: str) -> int:
    c = conn()
    n = c.execute("DELETE FROM users WHERE id=?", (uid,)).rowcount
    c.commit()
    return n


# ── OAuth 授权码 ──────────────────────────────────────────
def put_code(user_id: str, client_id: str, redirect_uri: str,
             challenge: str) -> str:
    code = secrets.token_urlsafe(32)
    c = conn()
    c.execute("""INSERT INTO oauth_codes(code,user_id,client_id,redirect_uri,
                 challenge,created) VALUES(?,?,?,?,?,?)""",
              (code, user_id, client_id, redirect_uri, challenge, time.time()))
    c.commit()
    return code


def take_code(code: str) -> Optional[Dict]:
    """一次性取用：取出即删。授权码重放是 OAuth 的经典攻击面。"""
    c = conn()
    r = c.execute("SELECT * FROM oauth_codes WHERE code=?", (code,)).fetchone()
    if not r:
        return None
    c.execute("DELETE FROM oauth_codes WHERE code=?", (code,))
    c.execute("DELETE FROM oauth_codes WHERE created < ?", (time.time() - 600,))
    c.commit()
    d = dict(r)
    return None if time.time() - d["created"] > 600 else d      # 10 分钟过期


# ── 视频 ──────────────────────────────────────────────────
def add_video(user_id: str, meta: Dict) -> Dict:
    """同一用户重复上传同一文件时复用已有记录（按指纹去重）。"""
    c = conn()
    old = c.execute("SELECT * FROM videos WHERE user_id=? AND fp=?",
                    (user_id, meta["fp"])).fetchone()
    if old:
        return dict(old)
    vid = secrets.token_hex(8)
    c.execute("""INSERT INTO videos(id,user_id,path,name,fp,duration,width,height,
                 quality,note,note_en,created,shot_at)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (vid, user_id, meta["path"], meta["name"], meta["fp"], meta["duration"],
               meta["width"], meta["height"], meta["quality"], meta.get("note", ""),
               meta.get("note_en", ""), time.time(), meta.get("shot_at")))
    c.commit()
    return dict(c.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone())


def video_by_fp(user_id: str, fp: str) -> Optional[Dict]:
    """按指纹找已有记录。上传时先查它，命中就不用再跑乒乓球判定（要 3.5 秒）。"""
    r = conn().execute("SELECT * FROM videos WHERE user_id=? AND fp=?",
                       (user_id, fp)).fetchone()
    return dict(r) if r else None


def set_shot_at(vid: str, ts: float) -> None:
    c = conn()
    c.execute("UPDATE videos SET shot_at=? WHERE id=?", (ts, vid))
    c.commit()


def set_note(vid: str, note: str, note_en: str, quality: int) -> None:
    c = conn()
    c.execute("UPDATE videos SET note=?, note_en=?, quality=? WHERE id=?",
              (note, note_en, quality, vid))
    c.commit()


def list_videos(user_id: str) -> List[Dict]:
    return [dict(r) for r in conn().execute(
        "SELECT * FROM videos WHERE user_id=? ORDER BY created DESC",
        (user_id,)).fetchall()]


def get_video(user_id: str, vid: str) -> Optional[Dict]:
    """带 user_id 查询 —— 归属校验和取数据是同一次操作，不给「忘了检查」留机会。"""
    return _row(conn().execute("SELECT * FROM videos WHERE id=? AND user_id=?",
                               (vid, user_id)).fetchone())


def video_by_path(user_id: str, path: str) -> Optional[Dict]:
    return _row(conn().execute("SELECT * FROM videos WHERE user_id=? AND path=?",
                               (user_id, path)).fetchone())


# ── 任务 ──────────────────────────────────────────────────
def create_job(user_id: str, video_id: str) -> str:
    jid = secrets.token_hex(8)
    c = conn()
    c.execute("""INSERT INTO jobs(id,user_id,video_id,state,stage,payload,created,updated)
                 VALUES(?,?,?,?,?,?,?,?)""",
              (jid, user_id, video_id, "running", "排队中", "{}",
               time.time(), time.time()))
    c.commit()
    return jid


def update_job(jid: str, **fields) -> None:
    payload = fields.pop("payload", None)
    sets, vals = [], []
    for k, v in fields.items():
        sets.append("%s=?" % k); vals.append(v)
    if payload is not None:
        sets.append("payload=?"); vals.append(json.dumps(payload, ensure_ascii=False))
    sets.append("updated=?"); vals.append(time.time())
    vals.append(jid)
    c = conn()
    c.execute("UPDATE jobs SET %s WHERE id=?" % ",".join(sets), vals)
    c.commit()


def get_job(user_id: str, jid: str) -> Optional[Dict]:
    r = conn().execute("SELECT * FROM jobs WHERE id=? AND user_id=?",
                       (jid, user_id)).fetchone()
    if not r:
        return None
    d = dict(r)
    d.update(json.loads(d.pop("payload") or "{}"))
    return d


def last_done_job(user_id: str, video_id: str) -> Optional[Dict]:
    r = conn().execute(
        """SELECT * FROM jobs WHERE user_id=? AND video_id=? AND state='done'
           ORDER BY updated DESC LIMIT 1""", (user_id, video_id)).fetchone()
    if not r:
        return None
    d = dict(r)
    d.update(json.loads(d.pop("payload") or "{}"))
    return d


def orphan_running_jobs() -> int:
    """进程重启后，之前「运行中」的任务不可能再有人推进它们 —— 标记为中断，
    否则前端会永远转圈等一个已经不存在的任务。"""
    c = conn()
    n = c.execute("""UPDATE jobs SET state='error', stage='服务重启，任务已中断'
                     WHERE state='running'""").rowcount
    c.commit()
    return n

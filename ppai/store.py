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

DB_PATH = os.environ.get("PIPO_DB", "pipo.db")
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
        _local.conn = c
    return c


def _row(r) -> Optional[Dict]:
    return dict(r) if r is not None else None


# ── 用户 ──────────────────────────────────────────────────
def create_user(name: str, is_admin: bool = False) -> Dict:
    uid, tok = secrets.token_hex(8), secrets.token_urlsafe(24)
    c = conn()
    c.execute("INSERT INTO users(id,name,token,is_admin,created) VALUES(?,?,?,?,?)",
              (uid, name, tok, int(is_admin), time.time()))
    c.commit()
    return {"id": uid, "name": name, "token": tok, "is_admin": is_admin}


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
                 quality,note,note_en,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
              (vid, user_id, meta["path"], meta["name"], meta["fp"], meta["duration"],
               meta["width"], meta["height"], meta["quality"], meta.get("note", ""),
               meta.get("note_en", ""), time.time()))
    c.commit()
    return dict(c.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone())


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

/**
 * 反馈收集端 —— Cloudflare Worker + R2。
 *
 * 客户端（ppai/contrib.py）发过来的是一个 npz：候选时刻 + 2048 维嵌入 +
 * 一段 JSON 元信息。没有音频、没有画面、没有文件名、没有路径。
 *
 * 这个 Worker 只做四件事，故意做得很少
 * ------------------------------------
 *  1. 校验体积和 schema —— 挡住误发和乱发
 *  2. 按安装 id + 日期存进 R2
 *  3. 按 IP 限频
 *  4. 什么都不回传给客户端，除了成功与否
 *
 * **不做的事，以及为什么：**
 *  - 不解析 npz。解析等于在边缘跑不受信任的输入解压，收益是零 ——
 *    真正的校验在离线训练时做（labels.audit 那一套）。
 *  - 不记 IP。限频用的是 IP 的哈希，写进 KV 且一天过期；原始 IP 不落盘。
 *    这份数据的全部价值在声学向量上，IP 只会变成负债。
 *  - 不做鉴权。客户端是买断制的桌面软件，没有账号体系；加 token 只会
 *    变成一个人人可从二进制里抠出来的常量，安全上是自欺。
 *    真正的防线是体积上限 + 限频 + 离线校验。
 *
 * 部署：
 *   npx wrangler deploy
 * 需要在 wrangler.toml 里绑定：
 *   R2 bucket  PIPO_FEEDBACK
 *   KV         PIPO_RATE
 */

const MAX_BYTES = 8 * 1024 * 1024;   // 一次上传的上限。实测一份反馈 92-327KB，
                                     // 8MB 足够几十段，同时挡住明显的滥用
const SCHEMA = "1";
const PER_DAY = 20;                  // 每个 IP 每天允许的上传次数

export default {
  async fetch(request, env) {
    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, X-Pipo-Install, X-Pipo-Schema",
      "Access-Control-Max-Age": "86400",
    };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });
    if (request.method !== "POST") {
      return json({ error: "只接受 POST" }, 405, cors);
    }

    const schema = request.headers.get("X-Pipo-Schema");
    if (schema !== SCHEMA) {
      // 明确告诉客户端是版本问题，而不是笼统的 400 ——
      // 老版本客户端应该知道自己该升级，而不是无声地重试
      return json({ error: `schema ${schema} 不受支持，当前是 ${SCHEMA}` }, 409, cors);
    }

    const install = (request.headers.get("X-Pipo-Install") || "").trim();
    if (!/^[0-9a-f]{8,64}$/.test(install)) {
      return json({ error: "缺少或格式错误的安装编号" }, 400, cors);
    }

    const len = Number(request.headers.get("Content-Length") || 0);
    if (len > MAX_BYTES) {
      return json({ error: `超过 ${MAX_BYTES / 1048576} MB 上限` }, 413, cors);
    }

    // 限频。**存的是 IP 的哈希，不是 IP** —— 这份数据的价值全在声学向量上，
    // 留着原始 IP 只会把一份匿名数据变成一份可关联的数据。
    const ip = request.headers.get("CF-Connecting-IP") || "0.0.0.0";
    const day = new Date().toISOString().slice(0, 10);
    const key = `rate:${day}:${await sha256hex(ip + "|" + day)}`;
    const seen = Number((await env.PIPO_RATE.get(key)) || 0);
    if (seen >= PER_DAY) {
      return json({ error: "今天上传得有点多了" }, 429, cors);
    }

    const body = await request.arrayBuffer();
    if (body.byteLength === 0) return json({ error: "空请求" }, 400, cors);
    if (body.byteLength > MAX_BYTES) {
      // Content-Length 是客户端给的、可以撒谎，所以读完再查一次
      return json({ error: "超过上限" }, 413, cors);
    }
    // npz 就是 zip，magic 必须是 PK\x03\x04。挡住明显不是这个格式的东西，
    // 但不解压 —— 在边缘解压不受信任的输入没有收益。
    const head = new Uint8Array(body.slice(0, 4));
    if (!(head[0] === 0x50 && head[1] === 0x4b && head[2] === 0x03 && head[3] === 0x04)) {
      return json({ error: "不是预期的格式" }, 400, cors);
    }

    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const path = `v${SCHEMA}/${day}/${install}/${stamp}.npz`;
    await env.PIPO_FEEDBACK.put(path, body, {
      httpMetadata: { contentType: "application/octet-stream" },
      customMetadata: { install, schema: SCHEMA, bytes: String(body.byteLength) },
    });
    await env.PIPO_RATE.put(key, String(seen + 1), { expirationTtl: 90000 });

    return json({ ok: true, bytes: body.byteLength }, 200, cors);
  },
};

function json(obj, status, extra) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", ...extra },
  });
}

async function sha256hex(s) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

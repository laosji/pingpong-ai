# 反馈收集端

客户端（`ppai/contrib.py`）默认**关闭**，用户在设置里明确开启后才会发送。
发送的是候选时刻 + 2048 维声学向量，没有音频、画面、文件名、路径。

## 部署

```bash
npx wrangler r2 bucket create pipo-feedback
npx wrangler kv namespace create PIPO_RATE     # 把返回的 id 填进 wrangler.toml
npx wrangler deploy
```

部署后拿到的地址填给客户端：

```bash
PIPO_CONTRIB_URL=https://pipo-feedback.<你的子域>.workers.dev
```

**没配这个环境变量时客户端不会发送任何东西**（`/api/contrib/send` 返回 503），
这是有意的：不给一个默认地址，免得哪天误开了就有数据往外走。

## 取回数据训练

```bash
npx wrangler r2 object get pipo-feedback/v1/2026-08-01/... --file out.npz
```

npz 里是 `meta`（JSON，含条目 id、秒数、候选数）加上每条的 `t{i}` / `f{i}`。
入训练集前仍然要过 `labels.audit` 那套判据 —— 收集端故意不做校验，
在边缘解压不受信任的输入没有收益。

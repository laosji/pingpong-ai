# 部署

## 一、开服务器

**2 核 / 4GB / 100GB SSD** 就够。实测：单任务峰值内存 349MB，一小时视频
分析约 48 秒，纯 CPU 不需要 GPU。

**地域比配置重要。** 这个产品的核心动作是上传几百 MB 视频，压缩比再好也
改变不了这一点。用户在国内就别放美国。

| 选项 | 权衡 |
|---|---|
| 阿里云/腾讯云 **香港** | 免 ICP 备案，能立刻上线。内测首选 |
| 阿里云/腾讯云 **国内** | 上传最快，但域名要备案（几周） |
| Hetzner / DigitalOcean | 便宜，只适合面向海外 |

## 二、部署

```bash
# 服务器上
git clone <repo> pipo && cd pipo
echo "PIPO_DOMAIN=pipo.example.com" > .env    # 先把域名 A 记录指到这台机器
docker compose up -d --build
```

首次启动会下载 PANNs 权重（312MB），下载到挂卷里，之后重建镜像不再重下。
Caddy 会自动申请 HTTPS 证书 —— 前提是域名已解析到本机且 80/443 通。

```bash
docker compose logs -f app      # 看初始化和任务日志
```

## 三、开通用户

```bash
docker compose exec app python -m ppai.users add 张三
# → https://pipo.example.com/enter?t=xxxxxxxx   把这条发给本人
docker compose exec app python -m ppai.users list
```

魔法链接点一次就种 cookie（90 天），之后不用再管。

## 四、要知道的几件事

**改上传上限要改两处。** `docker-compose.yml` 的 `PIPO_MAX_UPLOAD_MB` 和
`Caddyfile` 的 `max_size` 必须一致。不一致时小的那个先拒绝，用户看到的是
Caddy 的 413 错误页而不是我们的中文提示。

**反向代理超时不能用默认值。** 一小时视频分析 48 秒、加转码更久，默认超时
会在任务跑完前掐断连接。Caddyfile 里已设成 600s。

**数据都在 `pipo-data` 卷里**：数据库、标注、模型、上传、成片、缓存。
备份就是备份这个卷：

```bash
docker run --rm -v pipo_pipo-data:/d -v $(pwd):/b alpine \
  tar czf /b/pipo-backup-$(date +%F).tgz -C /d .
```

**真正该备份的只有 `labels/` 和 `models/`** —— 标注和固化嵌入是唯一
不可再生的资产（视频删了也能训练，见 README 第 17 节），成片和上传都是
到期即删的。

## 五、还没做的

- **速率限制**：没有。内测熟人规模可以先不做，公开前必须加。
- **对象存储**：视频现在存在服务器本地卷。放大后应该挪到 R2/OSS，
  服务器只留数据库和临时文件。
- **多机**：SQLite 决定了当前只能单机。真到需要横向扩展时换 Postgres，
  `ppai/store.py` 的接口不用动。

## 六、本文件未经验证

上面这套 Docker 配置**没有在真实环境跑过** —— 开发机上没有 Docker。
语法层面校验过（entrypoint 的 sh -n、compose 的 YAML 解析），
但镜像能不能构建成功、Caddy 能不能正常签证书，要到服务器上跑第一次才知道。

第一次部署建议盯着 `docker compose logs -f`，最可能出问题的三处：
torch CPU 源是否可达、PANNs 权重下载是否被墙、域名解析是否已生效。

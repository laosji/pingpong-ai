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

## 七、接入 ChatGPT / Claude

三种接法，覆盖不同客户端：

| 客户端 | 接法 | 用户要做什么 |
|---|---|---|
| **claude.ai 连接器** | `https://域名/mcp` | 粘贴 URL → 点「同意」 |
| **ChatGPT 连接器 / GPT Actions** | 同上，或 `/openapi.json` | 同上 |
| Claude Desktop / Code | `mcp_server.py`（stdio） | 装 Python、配 json |

远程连接器不用装任何东西，是主推方式。

### OAuth：用户不用碰 token

**先澄清一点**：OpenAI 和 Anthropic 都没有面向第三方的公开身份登录，
所以做不到「用 ChatGPT/Claude 账号识别用户」。身份始终是我们自己的（邀请码）。

但 MCP 规范里的 OAuth 能达到同样的体验 —— 助手把用户弹到**我们的**授权页，
点一次同意，token 自动回传，用户全程不碰 token：

```
claude.ai 添加连接器 https://域名/mcp
   → 401 + WWW-Authenticate 指回资源元数据
   → 客户端读 /.well-known/oauth-authorization-server
   → 动态注册（RFC 7591）拿 client_id
   → 弹出我们的授权页，用户填邀请码（或已登录则直接确认）
   → 回调带 code → 用 PKCE 换 access_token
   → 之后所有调用带 Bearer
```

全流程已实测跑通，四项安全检查通过：授权码不可重放、PKCE 强制校验、
无效邀请码被拒、危险回调地址被拒。

### 四个工具

| 工具 | 作用 |
|---|---|
| `pipo_upload_link` | 返回已带登录的上传链接（远程连接器用） |
| `pipo_add_local_video` | 读本地文件上传（仅 stdio 版可用） |
| `pipo_list_videos` | 列出已上传的录像 |
| `pipo_add_video` | 按公开直链添加录像 |
| `pipo_make_highlight` | 生成集锦，返回下载链接 |

### 本地文件：看 MCP server 跑在谁的机器上

| 客户端 | server 位置 | 本地文件 |
|---|---|---|
| Claude Desktop / Claude Code | **用户自己机器上**（stdio） | **能直接读** |
| claude.ai / ChatGPT 网页 | 我们的服务器 | 读不到 |

stdio 版是在用户机器上跑的进程，有文件系统权限，
所以 `pipo_add_local_video` 能直接读磁盘上的视频、流式上传
（不把文件读进内存，内存占用与文件大小无关）。

实测一次完整对话：

```
用户：帮我剪一下 ~/Downloads/训练.mp4 里最帅的球
  1. pipo_add_local_video  -> 读本地文件，304 秒的录像
  2. pipo_make_highlight   -> 最帅击球：5 段 / 20.2 秒
  3. 返回下载链接 + 结构分析（53 个回合，有效打球 42.4 秒）
```

远程连接器读不到用户磁盘，只能用 `pipo_upload_link` 给一条已带登录的
上传链接，用户在浏览器传完回来说一声，助手再 `pipo_list_videos` 取到。
同一件事在两种传输下答案不同，别混为一谈。

### Claude Desktop / Code（本地 stdio）

```json
{
  "mcpServers": {
    "pipo-ai": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "PIPO_BASE_URL": "https://pipo.example.com",
        "PIPO_TOKEN": "该用户魔法链接里的那串 token"
      }
    }
  }
}
```

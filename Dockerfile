# ── 构建镜像 ──────────────────────────────────────────────
# 关键：torch 必须走 CPU 源。Linux 上默认的 pip install torch 会拉 CUDA 依赖，
# 镜像从 ~1.5GB 涨到 5GB+，而我们全程 CPU 推理（实测一小时视频 48 秒），
# 根本用不到 GPU。
FROM python:3.13-slim

# fonts-noto-cjk 是叠字用的。没有它 render.caption_font() 返回 None，
# 代码会跳过叠字而不是崩，但成片上就不会有「第 N 回合」。约 60MB。
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg curl ca-certificates fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖再拷代码：代码改动不会让依赖层失效，重建快得多
COPY requirements.txt .
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt \
 && pip install --no-cache-dir "fastapi" "uvicorn[standard]" python-multipart

COPY ppai/ ./ppai/
COPY web/ ./web/
COPY server.py config.yaml ./
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 不用 root 跑。数据目录的属主在 entrypoint 里按挂载情况修正。
RUN useradd -m -u 10001 pipo && mkdir -p /data && chown -R pipo:pipo /app /data
USER pipo

# 模型权重和所有可变数据都在 /data（挂卷），镜像本身无状态
ENV PIPO_DB=/data/pipo.db \
    PIPO_DATA=/data \
    HOME=/data \
    PYTHONUNBUFFERED=1

EXPOSE 8020
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD curl -fsS http://127.0.0.1:8020/login.html >/dev/null || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8020"]

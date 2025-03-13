# 阶段 1：构建环境
FROM --platform=$BUILDPLATFORM python:3.10-slim AS builder

WORKDIR /src

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential swig zlib1g-dev libsdl2-dev ninja-build curl tree\
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -I \
    && pip install jupyterlab

# 安装 Tini
ENV TINI_VERSION=v0.6.0
RUN curl -fsSL https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini -o /usr/bin/tini \
    && chmod +x /usr/bin/tini

# 阶段 2：运行时环境
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /src

# 从构建阶段复制必要内容
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/bin/tini /usr/bin/tini
COPY . .

# 设置入口点
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=", "--NotebookApp.password=", "--allow-root"]
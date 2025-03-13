# 使用 Python 3.10 官方镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /src

# 安装系统依赖（如有必要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -I && \
    pip install jupyterlab

# 添加 Tini（进程管理器）
ENV TINI_VERSION v0.6.0
RUN curl -fsSL https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini -o /usr/bin/tini \
    && chmod +x /usr/bin/tini

# 复制项目代码
COPY . .

# 设置入口点
ENTRYPOINT ["/usr/bin/tini", "--"]

# 启动 Jupyter Lab
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
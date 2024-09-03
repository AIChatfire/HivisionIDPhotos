FROM ubuntu:22.04
# 环境变量
ENV HF_ENDPOINT=https://hf.chatfire.cc
ENV HF_TOKEN=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
# FEISHU_APP_SECRET
# MINIO_ACCESS_KEY

ENV OPENAI_BASE_URL=https://api.chatfire.cn/v1
ENV DIFY_BASE_URL=http://flow.chatfire.cn/v1

ENV SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
ENV MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
ENV DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
ENV ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ENV GROQ_BASE_URL=https://api.groq.com/openai/v1

# apt换源，安装pip
#RUN echo "==> 换成清华源，并更新..."  && \
#    sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list  && \
#    sed -i s@/security.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list  && \
#    apt-get clean  && \
#    apt-get update

# 安装python3.10
#RUN apt-get install -y python3 curl && \
#    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py  && \
#    python3 get-pip.py && \
#    pip3 install -U pip && \
#    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get clean && apt-get update && apt-get install -y python3 curl && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py  && \
    python3 get-pip.py && \
    pip3 install -U pip

# 安装ffmpeg等库
RUN apt-get install libpython3.10-dev ffmpeg libgl1-mesa-glx libglib2.0-0 cmake -y && \
    pip3 install --no-cache-dir cmake

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

RUN echo "==> Clean up..."  && \
    rm -rf ~/.cache/pip

# 指定工作目录

EXPOSE 7860

#CMD [ "python3", "app.py", "--host", "0.0.0.0", "--port", "7860"]

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

CMD ["sh", "-c", "python3 -m meutils.clis.server gunicorn-run main:app --port 8000 --workers ${WORKERS:-1} --threads 2 --pythonpath python3"]

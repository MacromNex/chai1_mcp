FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir chai_lab==0.6.1
RUN pip install --no-cache-dir --ignore-installed fastmcp
RUN pip install --no-cache-dir -U cryptography certifi

COPY src/ ./src/
RUN chmod -R a+r /app/src/
COPY scripts/ ./scripts/
RUN chmod -R a+r /app/scripts/
COPY configs/ ./configs/
RUN chmod -R a+r /app/configs/

RUN mkdir -p jobs/active jobs/completed jobs/logs results tmp

ENV PYTHONPATH=/app

CMD ["python", "src/server.py"]

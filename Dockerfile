FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install "pip==23.0.1" && \
    pip install --default-timeout=1000 "setuptools==65.5.0" "wheel<0.40.0" && \
    pip install --default-timeout=10000 --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .

ENV PYTHONPATH="${PYTHONPATH}:/app/src"
ENV CUDA_VISIBLE_DEVICES=""

CMD ["bash"]
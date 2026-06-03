FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# PyTorch CPU — explicit index avoids downloading the multi-GB CUDA wheel
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# NOTE — open-cd (ChangeFormer) skipped here.
# openmim pulls aliyun/openxlab which pins setuptools~=60.x, broken on Python 3.12.
# mmcv/mmsegmentation/open-cd are installed inside the container in Phase 4
# using: pip install mmengine mmcv mmsegmentation open-cd (direct PyPI, no openmim).

# GDAL Python bindings must match the system gdal-bin version exactly.
# Project deps cached here — layer invalidated only when pyproject.toml changes.
COPY pyproject.toml .
RUN mkdir -p app && touch app/__init__.py && \
    pip install --no-cache-dir "GDAL==$(gdal-config --version)" && \
    pip install --no-cache-dir -e .

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

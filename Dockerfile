# Python-based 6-DoF Relative Pose Estimation
# Multi-platform support: linux/amd64, linux/arm64

ARG PLATFORM=linux/amd64
ARG PYTHON_VERSION=3.9
ARG ENTRY_FILE=""

FROM --platform=$PLATFORM python:${PYTHON_VERSION}-slim

ARG TARGETPLATFORM
ARG ENTRY_FILE
ENV DEBIAN_FRONTEND=noninteractive
ENV ENTRY_FILE=${ENTRY_FILE}

LABEL maintainer="6-DoF Pose Estimation Project"
LABEL description="Python-based pose estimation"
LABEL target.platform="${TARGETPLATFORM}"

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

ENV PYTHONPATH=/app

CMD ["/bin/sh", "-c", "if [ -n \"$ENTRY_FILE\" ]; then MODULE=$(echo \"$ENTRY_FILE\" | sed 's/\\.py$//' | sed 's/\\//./g'); python3 -m \"$MODULE\"; else python3; fi"]

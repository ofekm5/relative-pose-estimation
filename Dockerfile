# ============================================================================
# Dockerfile for 6-DoF Relative Pose Estimation (Python Implementation)
# ============================================================================
# Supports: WSL (linux/amd64), Ubuntu (linux/amd64), Raspberry Pi 4 (linux/arm64)
#
# Build commands:
#   docker build -t pose-estimator .
#   make docker-build
#
# Run examples:
#   # Interactive Python shell with mounted data
#   docker run -it --rm -v $(pwd)/silmulator_data:/app/silmulator_data pose-estimator bash
#
#   # Run evaluation with visualization
#   docker run --rm -v $(pwd)/silmulator_data:/app/silmulator_data \
#              -v $(pwd)/results:/app/results pose-estimator
#
# This is a Python-only implementation for university coursework
# ============================================================================

ARG PLATFORM=linux/amd64
ARG PYTHON_VERSION=3.9
ARG ENTRY_FILE=""

################################################################################
# Runtime Image - Python with OpenCV and dependencies
################################################################################
FROM --platform=$PLATFORM python:${PYTHON_VERSION}-slim

ARG TARGETPLATFORM
ARG ENTRY_FILE
ENV DEBIAN_FRONTEND=noninteractive
ENV ENTRY_FILE=${ENTRY_FILE}

# Add metadata labels
LABEL maintainer="6-DoF Pose Estimation University Project"
LABEL description="Python-based pose estimation for WSL, Ubuntu (amd64) and Raspberry Pi 4 (arm64)"
LABEL target.platform="${TARGETPLATFORM}"

# ============================================================================
# Install System Dependencies for OpenCV
# ============================================================================
# libgl1: OpenGL support for cv2 (required by opencv-python)
# libglib2.0-0: GLib library (required by cv2)
# libsm6, libxext6, libxrender1: X11 libraries (required by cv2)
# libgomp1: GNU OpenMP library for parallel processing
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# ============================================================================
# Install Python Dependencies
# ============================================================================
# opencv-python: OpenCV with pre-built binaries (faster than building from source)
# numpy: Numerical computing
# pandas: Data manipulation and analysis
# plotly: Interactive 3D visualization
# matplotlib: Plotting and video generation
# ============================================================================
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source code
COPY src/ ./src/

# Copy test scripts
COPY tests/ ./tests/

# Copy data directory structure (optional, can be mounted at runtime)
# COPY silmulator_data/ ./silmulator_data/

# Set Python path to include src directory
ENV PYTHONPATH=/app:$PYTHONPATH

# ============================================================================
# Container Execution Configuration
# ============================================================================
# Default behavior: Run Python interactive shell or specified entry file
# Users can override with specific Python scripts or bash for debugging
#
# Build with entry file:
#   docker build --build-arg ENTRY_FILE=src/plots_graths.py -t pose-estimator .
#
# Run examples:
#   docker run -it pose-estimator                           # Python shell or entry file
#   docker run -it pose-estimator bash                      # Bash shell
#   docker run pose-estimator python -c "from src.pose_matcher import PoseMatcher; ..."
# ============================================================================
CMD if [ -n "$ENTRY_FILE" ]; then python3 $ENTRY_FILE; else python3; fi

# Build Guide

This document explains how to set up and run the 6-DoF Relative Pose Estimation system (Python implementation).

## Quick Start

```bash
# Show all available commands
make help

# Install Python dependencies locally
make install

# Run evaluation with visualization
make run-eval

# Generate 3D plots
make run-plot

# Generate video with ground truth overlay
make run-video

# Docker build
make docker-build

# Docker run (interactive)
make docker-run
```

## Build System Overview

The project uses **Python 3** with **pip** for dependency management and a **Makefile wrapper** for convenience.

### Two-Layer Build System

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Makefile (Convenience Layer)                            │
│    - Shortcuts: make install, make run-eval, make docker   │
│    - Calls pip or Python commands                          │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
┌──────────────┐    ┌─────────────────────────────────────┐
│ 2a. pip      │    │ 2b. Dockerfile (single stage)      │
│ (Local)      │    │     - Python 3.9 base image         │
│              │    │     - pip install requirements      │
│              │    │     - Copy Python source            │
└──────────────┘    └─────────────────────────────────────┘
```

### Directory Structure

```
relative-pose-estimation/
├── requirements.txt        # Python dependencies
├── Makefile                # Convenience wrapper
├── Dockerfile              # Docker container build
├── src/                    # Python source files (.py)
│   ├── pose_matcher.py    # Main pose estimation pipeline
│   ├── plots_graths.py    # Visualization and video generation
│   ├── feature_extractor.py
│   ├── matcher.py
│   ├── image_loader.py
│   ├── gt_utils.py
│   └── ...                # Additional utilities
└── silmulator_data/        # Ground truth data and test images
    └── simple_movement/
        ├── camera_poses.txt
        └── images/
```

---

## 📘 Understanding requirements.txt

### What is requirements.txt?

`requirements.txt` is a **standard Python convention** for specifying project dependencies. It lists all Python packages needed to run the project, along with their version constraints.

### How requirements.txt Works

```txt
# Computer Vision
opencv-python>=4.8.0        # OpenCV with pre-built binaries

# Numerical Computing
numpy>=1.24.0               # Array operations and linear algebra

# Data Processing
pandas>=2.0.0               # Ground truth data loading and manipulation

# Visualization
plotly>=5.14.0              # Interactive 3D trajectory plots
matplotlib>=3.7.0           # Video generation and 2D plotting
```

### Where requirements.txt is Used

**1. Local Installation**
```bash
pip install -r requirements.txt
```

**2. Inside Docker (Dockerfile)**
```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

**Key Feature:** Pin major versions with `>=` to ensure compatibility while allowing minor updates!

---

## 🐳 Understanding Python Docker Build

### Why Python Docker is Simpler

**Python with Pre-built Wheels:**
```
Final Image Size: ~500 MB
Build Time: 2-3 minutes
Contains: Python runtime + pip packages + source code
```

**Advantages over C++ Multi-Stage Build:**
- ✅ No compilation needed (opencv-python has pre-built binaries)
- ✅ Faster builds (pip downloads wheels instead of compiling)
- ✅ Simpler Dockerfile (single stage instead of 3)
- ✅ Easier to debug (can install packages interactively)

### The Single-Stage Build Explained

```dockerfile
FROM python:3.9-slim

# Install system dependencies (required by opencv-python)
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path and default command
ENV PYTHONPATH=/app:$PYTHONPATH
CMD ["python3"]
```

**What it does:**
1. Starts with official Python 3.9 slim image (~150 MB)
2. Installs system libraries needed by OpenCV (libgl1, etc.)
3. Installs Python packages via pip (~300 MB for opencv-python + deps)
4. Copies Python source code (~1 MB)
5. Sets up environment and entry point

### Docker Layer Caching

```
First Build:
  Install system deps          → 30 seconds
  Install Python packages      → 90 seconds
  Copy source code             → 1 second

After Editing src/pose_matcher.py:
  Install system deps          → ✅ CACHED (0 seconds)
  Install Python packages      → ✅ CACHED (0 seconds)
  Copy source code             → 1 second (rebuilds this layer only)
```

Docker caches layers efficiently. Changing source code doesn't reinstall dependencies!

---

## 🔧 Understanding Dockerfile ARG Variables

### ARG vs ENV

| Type | Scope | When Set | Example |
|------|-------|----------|---------|
| `ARG` | Build-time only | Before/during `docker build` | Python version, entry file |
| `ENV` | Build + Runtime | Persists in running container | PYTHONPATH, ENTRY_FILE |

### Where ARGs Are Set

#### **1. ARG Defined in Dockerfile (with defaults)**

```dockerfile
ARG PLATFORM=linux/amd64        # Default: x86_64
ARG PYTHON_VERSION=3.9          # Default: Python 3.9
ARG ENTRY_FILE=""               # Default: empty (interactive shell)
```

These are **default values**. You can override them during build.

---

#### **2. Overriding ARGs at Build Time**

```bash
# Build with Python 3.10
docker build --build-arg PYTHON_VERSION=3.10 -t pose-estimator .

# Build with specific entry file
docker build --build-arg ENTRY_FILE=src/plots_graths.py -t pose-estimator .

# Build with multiple args
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg ENTRY_FILE=src/pose_matcher.py \
  -t pose-estimator .
```

---

#### **3. Using ENTRY_FILE in Docker**

```dockerfile
ARG ENTRY_FILE=""
ENV ENTRY_FILE=${ENTRY_FILE}

CMD if [ -n "$ENTRY_FILE" ]; then python3 $ENTRY_FILE; else python3; fi
```

**Behavior:**
- If `ENTRY_FILE` is empty → Starts Python interactive shell
- If `ENTRY_FILE` is set → Runs the specified Python script

---

### ARG Flow Diagram

```
┌──────────────────────────────────────────┐
│ docker build --build-arg ENTRY_FILE=...  │
└───────────┬──────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ Dockerfile ARG (defaults)                │
│ ARG PLATFORM=linux/amd64                 │
│ ARG PYTHON_VERSION=3.9                   │
│ ARG ENTRY_FILE=""                        │ ← Override here
└───────────┬──────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ Pass to ENV (persists in container)      │
│ ENV ENTRY_FILE=${ENTRY_FILE}             │
└───────────┬──────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ Use in CMD                               │
│ if [ -n "$ENTRY_FILE" ]; then ...        │ ← Run script or shell
└──────────────────────────────────────────┘
```

---

## Local Setup (Without Docker)

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation Steps

```bash
# Install dependencies
make install

# Or manually:
pip install -r requirements.txt
```

### Run Locally

```bash
# Run evaluation with visualization
make run-eval

# Or manually:
python -c "
from src.plots_graths import PosePlotter
plotter = PosePlotter('silmulator_data/simple_movement', step=15)
plotter.run()
"

# Run single frame pair
python -c "
from src.pose_matcher import PoseMatcher
from src.image_loader import load_image_pair

matcher = PoseMatcher('silmulator_data/simple_movement',
                      'silmulator_data/simple_movement/camera_poses.txt')
img1, img2 = load_image_pair(
    'silmulator_data/simple_movement/images/000000.png',
    'silmulator_data/simple_movement/images/000015.png',
    to_gray=True
)
yaw, pitch, roll = matcher.match(img1, img2, prev_frame_index=0)
print(f'Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°')
"

# Generate video
make run-video
```

### Clean Python Cache

```bash
make clean
```

## Docker Build

### Build Image

```bash
# Default: interactive Python shell
make docker-build

# With specific entry file
make docker-build ENTRY_FILE=src/plots_graths.py
```

This builds a Docker image named `pose-estimator:latest` with Python 3.9, all dependencies, and your source code.

### Run with Docker

```bash
# Interactive Python shell with mounted data
make docker-run

# Interactive bash shell
make docker-shell

# Run specific Python command
docker run --rm \
  -v $(pwd)/silmulator_data:/app/silmulator_data \
  -v $(pwd)/results:/app/results \
  pose-estimator:latest \
  python -c "from src.pose_matcher import PoseMatcher; print('Hello from Docker')"

# Run evaluation inside container
docker run --rm \
  -v $(pwd)/silmulator_data:/app/silmulator_data \
  -v $(pwd)/results:/app/results \
  pose-estimator:latest \
  python -c "from src.plots_graths import PosePlotter; \
             plotter = PosePlotter('silmulator_data/simple_movement', step=15); \
             plotter.run()"
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make help` | Show available commands |
| `make install` | Install Python dependencies locally |
| `make run-eval` | Run pose estimation evaluation |
| `make run-plot` | Generate 3D trajectory plots |
| `make run-video` | Generate video with GT overlay |
| `make clean` | Remove Python cache files |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container (interactive) |
| `make docker-shell` | Open bash shell in container |

## Platform Notes

### Windows

- Use Git Bash, WSL, or PowerShell for make commands
- Python 3.9+ required (download from python.org or Windows Store)
- Docker Desktop required for Docker builds

### Linux

- Install Python: `sudo apt install python3 python3-pip`
- System dependencies for opencv-python: `sudo apt install libgl1 libglib2.0-0`

### macOS

- Install via Homebrew: `brew install python@3.9`
- System dependencies usually pre-installed

### Raspberry Pi

- Python is pre-installed on Raspberry Pi OS
- opencv-python has ARM64 wheels (no compilation needed)
- For Docker builds:

```bash
# On PC: build ARM64 image
docker buildx build --platform linux/arm64 -t pose-estimator:latest .
docker save pose-estimator:latest | gzip > pose-estimator.tar.gz

# Transfer to Pi
scp pose-estimator.tar.gz pi@raspberrypi.local:~

# On Pi: load and run
docker load < pose-estimator.tar.gz
docker run --rm \
  -v ~/silmulator_data:/app/silmulator_data \
  pose-estimator:latest python -c "from src.pose_matcher import PoseMatcher; print('Works on Pi!')"
```

## Troubleshooting

### ImportError: No module named 'cv2'

**Error:** `ImportError: No module named 'cv2'`

**Solution:**
```bash
# Ensure opencv-python is installed
pip install opencv-python

# Or reinstall all dependencies
pip install -r requirements.txt
```

### OpenCV ImportError with Missing Libraries

**Error:** `ImportError: libGL.so.1: cannot open shared object file`

**Solution (Linux):**
```bash
sudo apt install libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

### Python Version Issues

**Error:** `SyntaxError` or version-related errors

**Solution:**
```bash
# Check Python version (requires 3.9+)
python --version

# Use specific Python version
python3.9 -m pip install -r requirements.txt
```

### Clean Installation

```bash
# Remove cache and reinstall
make clean
make install
```

### Docker Issues

```bash
# Rebuild without cache
docker build --no-cache -t pose-estimator:latest .

# Check Docker logs
docker run --rm -it pose-estimator:latest bash
python -c "import cv2; print(cv2.__version__)"
```

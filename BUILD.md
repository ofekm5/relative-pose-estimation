# Build Guide

This document explains how to build and run the 6-DoF Relative Pose Estimation system.

## Quick Start

```bash
# Show all available commands
make help

# Local build (requires OpenCV installed)
make build

# Run locally
make run IMG1=images/before.jpg IMG2=images/after.jpg

# Docker build
make docker-build

# Docker run
make docker-run
```

## Build System Overview

The project uses **CMake** for building C++ code and a simple **Makefile wrapper** for convenience.

### Directory Structure

```
relative-pose-estimation/
├── CMakeLists.txt          # CMake configuration
├── Makefile                # Simple wrapper for common commands
├── Dockerfile              # Docker container setup
├── include/                # Header files
├── src/                    # Source files (.c, .cpp)
│   ├── main.c             # Entry point
│   └── ...                # FeatureExtractor, PoseEstimator modules
└── images/                 # Test images
```

## Local Build (Without Docker)

### Prerequisites

- CMake 3.10 or higher
- C++ compiler with C++11 support
- OpenCV 4.x installed

### Build Steps

```bash
# Create build directory and compile
make build

# Or manually:
mkdir -p build
cd build
cmake ..
cmake --build .
```

This creates the executable at `build/pose_estimator`.

### Run Locally

```bash
# Using Makefile wrapper
make run IMG1=path/to/image1.jpg IMG2=path/to/image2.jpg

# Or directly
./build/pose_estimator path/to/image1.jpg path/to/image2.jpg
```

### Clean Build

```bash
make clean
```

## Docker Build

### Build Image

```bash
make docker-build
```

This builds a Docker image named `pose-estimator:latest` with OpenCV and the compiled executable.

### Run with Docker

```bash
# Default: uses images/before.jpg and images/after.jpg
make docker-run

# Custom images (place them in images/ directory)
docker run --rm -v $(pwd)/images:/data pose-estimator:latest /data/img1.jpg /data/img2.jpg
```

## Output Format

The program outputs:

```
Rotation matrix R:
[3x3 matrix]

Translation vector T:
[3x1 vector]

Roll: X.XX  Pitch: Y.YY  Yaw: Z.ZZ
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make help` | Show available commands |
| `make build` | Build locally using CMake |
| `make clean` | Remove build directory |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container |
| `make run IMG1=... IMG2=...` | Run locally built executable |

## Platform Notes

### Windows

- Use Git Bash, WSL, or PowerShell for make commands
- Ensure OpenCV is in your PATH for local builds
- Docker Desktop required for Docker builds

### Linux

- Install build essentials: `sudo apt install build-essential cmake`
- Install OpenCV: `sudo apt install libopencv-dev`

### macOS

- Install via Homebrew: `brew install cmake opencv`

### Raspberry Pi

- The Dockerfile includes ARM/NEON optimizations
- Build on a PC and transfer the image for faster builds:

```bash
# On PC: build and save
docker build -t pose-estimator:latest .
docker save pose-estimator:latest | gzip > pose-estimator.tar.gz

# Transfer to Pi
scp pose-estimator.tar.gz pi@raspberrypi.local:~

# On Pi: load and run
docker load < pose-estimator.tar.gz
docker run --rm -v ~/images:/data pose-estimator:latest /data/before.jpg /data/after.jpg
```

## Development Workflow

### Module Development (Partner A: FeatureExtractor)

```bash
# Edit src/FeatureExtractor.cpp and include/FeatureExtractor.h
# Rebuild
make build

# Test
make run IMG1=test1.jpg IMG2=test2.jpg
```

### Module Development (Partner B: PoseEstimator)

```bash
# Edit src/PoseEstimator.cpp and include/PoseEstimator.h
# Rebuild
make build

# Test
make run IMG1=test1.jpg IMG2=test2.jpg
```

### Integration Testing

```bash
# Build Docker image with both modules
make docker-build

# Run full pipeline
make docker-run
```

## Troubleshooting

### OpenCV Not Found

**Error:** `Could not find OpenCV`

**Solution:**
```bash
# Linux
sudo apt install libopencv-dev

# macOS
brew install opencv

# Windows
# Download from opencv.org and set OpenCV_DIR environment variable
```

### Build Fails

```bash
# Clean and rebuild
make clean
make build
```

### Docker Issues

```bash
# Rebuild without cache
docker build --no-cache -t pose-estimator:latest .
```

## Next Steps

- Add test images to `images/` directory
- Implement FeatureExtractor module (Partner A)
- Implement PoseEstimator module (Partner B)
- Test with real camera frames
- Deploy to Raspberry Pi

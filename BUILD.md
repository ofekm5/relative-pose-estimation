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

### Three-Layer Build System

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Makefile (Convenience Layer)                            │
│    - Shortcuts: make build, make docker-build, make run    │
│    - Just calls CMake or Docker commands                   │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
┌──────────────┐    ┌─────────────────────────────────────┐
│ 2a. CMake    │    │ 2b. Dockerfile (3 stages)          │
│ (Local)      │    │     - Uses CMake internally         │
│              │    │     - Stage 1: Build OpenCV         │
│              │    │     - Stage 2: Build project (CMake)│
│              │    │     - Stage 3: Runtime image        │
└──────────────┘    └─────────────────────────────────────┘
```

### Directory Structure

```
relative-pose-estimation/
├── CMakeLists.txt          # CMake configuration (used locally AND in Docker)
├── Makefile                # Convenience wrapper
├── Dockerfile              # Multi-stage container build
├── include/                # Header files (.h, .hpp)
├── src/                    # Source files (.c, .cpp)
│   ├── main.cpp           # Entry point (C++ file)
│   ├── MatchResult.h      # Shared data structures
│   └── ...                # FeatureExtractor, PoseEstimator modules
└── images/                 # Test images (mounted to /data in Docker)
```

---

## 📘 Understanding CMakeLists.txt

### What is CMake?

CMake is a **cross-platform build system generator**. It doesn't compile code directly—instead, it generates build files (Makefiles, Visual Studio projects, etc.) that compile your code.

### How CMakeLists.txt Works

```cmake
# 1. Project Setup
cmake_minimum_required(VERSION 3.10)
project(RelativePoseEstimation)

# 2. Compiler Settings
set(CMAKE_CXX_STANDARD 11)              # Use C++11 features
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # Enforce C++11 (don't fall back)

# 3. Platform Detection (NEW - auto-detects x86_64 vs ARM64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
    # Raspberry Pi 4 detected
    message(STATUS "ARM64 platform detected")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)")
    # WSL/Ubuntu detected
    message(STATUS "x86_64 platform detected")
endif()

# 4. Find Dependencies
find_package(OpenCV REQUIRED)           # Locate OpenCV library

# 5. Include Directories (where to find .h files)
include_directories(
    ${OpenCV_INCLUDE_DIRS}              # OpenCV headers
    ${CMAKE_SOURCE_DIR}/include         # Your headers
    ${CMAKE_SOURCE_DIR}/src
)

# 6. Collect Source Files (auto-discovers all .c and .cpp files)
file(GLOB SOURCES "src/*.c" "src/*.cpp")

# 7. Build Executable
add_executable(pose_estimator ${SOURCES})

# 8. Link Libraries
target_link_libraries(pose_estimator ${OpenCV_LIBS})

# 9. Install (for 'make install' - copies binary to /usr/local/bin)
install(TARGETS pose_estimator DESTINATION bin)
```

### Where CMakeLists.txt is Used

**1. Local Builds (Direct)**
```bash
mkdir build && cd build
cmake ..                    # ← CMakeLists.txt reads here
cmake --build .             # ← Compiles using generated Makefile
```

**2. Inside Docker (Dockerfile Stage 2)**
```dockerfile
# Stage 2: Project Builder
COPY CMakeLists.txt ./
RUN cmake -D CMAKE_BUILD_TYPE=Release ..  # ← CMakeLists.txt used here
RUN make -j4
```

**Key Feature:** CMakeLists.txt auto-discovers files in `src/`, so partners can add `.cpp` files without editing it!

---

## 🐳 Understanding Multi-Stage Docker Build

### Why Multi-Stage Build?

**Without Multi-Stage (Single-Stage Build):**
```
Final Image Size: ~2.5 GB
Contains: Source code, build tools, intermediate files, OpenCV source, docs, examples
```

**With Multi-Stage Build:**
```
Final Image Size: ~150 MB (94% smaller!)
Contains: Only compiled binary + runtime libraries
```

### The Three Stages Explained

#### **Stage 1: opencv-builder** (~1.5 GB intermediate image)
```dockerfile
FROM debian:bullseye-slim AS opencv-builder

# Download OpenCV 4.8.1 source (~50 MB)
RUN wget https://github.com/opencv/opencv/archive/4.8.1.zip

# Build OpenCV with minimal flags
RUN cmake -D BUILD_LIST=core,imgproc,features2d,calib3d ...
RUN make -j4
RUN make install
```

**What it does:**
- Downloads OpenCV source code
- Compiles OpenCV from scratch (~15-20 min)
- Only builds 6 essential modules (not all 20+)
- Platform detection: Uses SSE2 (x86) or NEON (ARM)
- **Result:** OpenCV libraries in `/usr/local/lib`

**Why needed?** OpenCV binary packages don't have optimal flags for our use case.

---

#### **Stage 2: project-builder** (~100 MB intermediate image)
```dockerfile
FROM debian:bullseye-slim AS project-builder

# Copy OpenCV from Stage 1 (no rebuild!)
COPY --from=opencv-builder /usr/local /usr/local

# Copy project source
COPY src/ ./src/
COPY include/ ./include/
COPY CMakeLists.txt ./

# Build project using CMakeLists.txt
RUN cmake -D CMAKE_BUILD_TYPE=Release ..
RUN make -j4
RUN make install
```

**What it does:**
- Reuses OpenCV from Stage 1 (via `COPY --from=opencv-builder`)
- Compiles your C++ code using CMakeLists.txt
- **Result:** `pose_estimator` binary in `/usr/local/bin`

**Why needed?** Separates OpenCV build (slow, rarely changes) from project build (fast, changes often).

---

#### **Stage 3: runtime** (~150 MB final image)
```dockerfile
FROM debian:bullseye-slim

# Copy ONLY what's needed to run (not build)
COPY --from=opencv-builder /usr/local/lib /usr/local/lib
COPY --from=project-builder /usr/local/bin/pose_estimator /usr/local/bin/

# Install minimal runtime dependencies (JPEG, PNG libraries)
RUN apt-get install libjpeg62-turbo libpng16-16

ENTRYPOINT ["/usr/local/bin/pose_estimator"]
```

**What it does:**
- Starts fresh with minimal Debian base
- Copies compiled OpenCV libraries (not source)
- Copies compiled binary (not source code)
- No build tools (gcc, cmake, wget) → smaller image

**Why needed?** Production image should only have what's needed to **run**, not **build**.

---

### Docker Layer Caching Magic

```
First Build:
  Stage 1: Build OpenCV        → 20 minutes
  Stage 2: Build project       → 30 seconds
  Stage 3: Create runtime      → 10 seconds

After Editing src/main.cpp:
  Stage 1: ✅ CACHED           → 0 seconds (unchanged)
  Stage 2: Rebuild             → 30 seconds (source changed)
  Stage 3: Rebuild             → 10 seconds (depends on Stage 2)
```

Docker caches layers until something changes. Changing `src/main.cpp` doesn't invalidate Stage 1!

---

## 🔧 Understanding Dockerfile ARG Variables

### ARG vs ENV

| Type | Scope | When Set | Example |
|------|-------|----------|---------|
| `ARG` | Build-time only | Before/during `docker build` | Platform detection |
| `ENV` | Build + Runtime | Persists in running container | Environment variables |

### Where ARGs Are Set

#### **1. ARG Defined in Dockerfile (with defaults)**

```dockerfile
# Line 7-8 in Dockerfile
ARG PLATFORM=linux/amd64        # Default: x86_64
ARG OPENCV_VERSION=4.8.1        # Default: OpenCV 4.8.1
```

These are **default values**. You can override them during build.

---

#### **2. ARG Auto-Set by Docker Buildx**

```dockerfile
# Line 15, 21, 128 in Dockerfile
ARG TARGETPLATFORM
```

**Where it's set:** Automatically by Docker based on your host platform

**Possible values:**
- `linux/amd64` → WSL, Ubuntu on x86_64
- `linux/arm64` → Raspberry Pi 4

**How Docker determines it:**
```bash
docker build .              # Auto-detects your host platform
docker buildx build --platform linux/arm64 .  # Override for cross-compilation
```

---

#### **3. Using ARG in Dockerfile**

```dockerfile
# Line 50-71 in Dockerfile
RUN case "$TARGETPLATFORM" in \
        "linux/amd64") \
            CMAKE_FLAGS="-D CPU_BASELINE=SSE2"; \
            BUILD_JOBS=4; \
            ;; \
        "linux/arm64") \
            CMAKE_FLAGS="-D ENABLE_NEON=ON"; \
            BUILD_JOBS=2; \
            ;; \
    esac
```

The `$TARGETPLATFORM` variable is used in shell script inside `RUN` to choose platform-specific flags.

---

### ARG Flow Diagram

```
┌──────────────────────────┐
│ docker build .           │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ Docker Buildx (auto-detects)             │
│ TARGETPLATFORM = "linux/amd64"           │ ← Set automatically
│ BUILDPLATFORM = "linux/amd64"            │
└───────────┬──────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ Dockerfile ARG (defaults)                │
│ ARG PLATFORM=linux/amd64                 │ ← Can override
│ ARG OPENCV_VERSION=4.8.1                 │ ← Can override
│ ARG TARGETPLATFORM                       │ ← From Docker
└───────────┬──────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ Use in RUN commands                      │
│ case "$TARGETPLATFORM" in ...            │ ← Read the value
│ CMAKE_FLAGS based on platform            │
└──────────────────────────────────────────┘
```

---

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

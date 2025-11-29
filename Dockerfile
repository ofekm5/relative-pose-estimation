# ============================================================================
# Multi-Platform Dockerfile for 6-DoF Relative Pose Estimation (MVP)
# ============================================================================
# Supports: WSL (linux/amd64), Ubuntu (linux/amd64), Raspberry Pi 4 (linux/arm64)
#
# Build commands:
#   make docker-build              # Build for current platform
#   docker build -t pose-estimator .
#
# This is a simplified MVP version for university coursework
# ============================================================================

ARG PLATFORM=linux/amd64
ARG OPENCV_VERSION=4.8.1

################################################################################
# Stage 1: OpenCV Builder - Compile OpenCV from source
################################################################################
FROM --platform=$PLATFORM debian:bullseye-slim AS opencv-builder

ARG TARGETPLATFORM
ARG OPENCV_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    unzip \
    ca-certificates \
    pkg-config \
    libjpeg62-turbo-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*
# Note: ca-certificates is required for wget to download from HTTPS (GitHub)

# Download OpenCV source code
WORKDIR /opencv_build
RUN wget -q -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip -q opencv.zip && \
    rm opencv.zip && \
    mkdir -p opencv-${OPENCV_VERSION}/build

WORKDIR /opencv_build/opencv-${OPENCV_VERSION}/build

# ============================================================================
# Platform Detection & Minimal Optimization Flags
# ============================================================================
# For WSL/Ubuntu (x86_64): Basic SSE2 support (available on all modern x86 CPUs)
# For Raspberry Pi 4 (ARM64): Enable NEON (ARM's SIMD instruction set)
# ============================================================================
RUN case "$TARGETPLATFORM" in \
        "linux/amd64") \
            # WSL & Ubuntu (x86_64) - Minimal optimizations
            # CPU_BASELINE=SSE2: Use SSE2 instructions (universal on x86_64)
            CMAKE_FLAGS="-D CPU_BASELINE=SSE2"; \
            BUILD_JOBS=4; \
            ;; \
        "linux/arm64") \
            # Raspberry Pi 4 (ARM64) - Minimal optimizations
            # ENABLE_NEON=ON: Use ARM NEON SIMD instructions
            CMAKE_FLAGS="-D ENABLE_NEON=ON"; \
            BUILD_JOBS=2; \
            ;; \
        *) \
            # Fallback for unknown platforms
            CMAKE_FLAGS=""; \
            BUILD_JOBS=2; \
            ;; \
    esac && \
    echo "$CMAKE_FLAGS" > /tmp/cmake_flags.txt && \
    echo "$BUILD_JOBS" > /tmp/build_jobs.txt

# ============================================================================
# Build OpenCV with Minimal Flags (MVP Configuration)
# ============================================================================
RUN CMAKE_FLAGS=$(cat /tmp/cmake_flags.txt) && \
    BUILD_JOBS=$(cat /tmp/build_jobs.txt) && \
    cmake \
    # Build type and install location
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    \
    # Only build essential OpenCV modules (reduces build time & image size)
    # core: Basic data structures
    # imgproc: Image processing
    # imgcodecs: Image file I/O (JPEG, PNG)
    # features2d: ORB feature detection
    # calib3d: Essential matrix, recoverPose
    # flann: Fast nearest neighbor search (for matching)
    -D BUILD_LIST=core,imgproc,imgcodecs,features2d,calib3d,flann \
    \
    # Disable unnecessary components (reduces build time)
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_apps=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_opencv_java=OFF \
    \
    # Disable GUI (not needed for headless Docker)
    -D WITH_GTK=OFF \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=OFF \
    \
    # Disable video/camera support (not needed for static images)
    -D WITH_GSTREAMER=OFF \
    -D WITH_FFMPEG=OFF \
    -D WITH_V4L=OFF \
    \
    # Disable Intel-specific libraries (not available on ARM, not needed for MVP)
    -D WITH_IPP=OFF \
    -D WITH_ITT=OFF \
    \
    # Enable ORB features (required for this project)
    -D OPENCV_ENABLE_NONFREE=ON \
    \
    # Build shared libraries to reduce final image size
    -D BUILD_SHARED_LIBS=ON \
    \
    # Platform-specific flags (SSE2 for x86, NEON for ARM)
    $CMAKE_FLAGS \
    .. && \
    echo "Building OpenCV for $TARGETPLATFORM with -j${BUILD_JOBS}" && \
    make -j${BUILD_JOBS} && \
    make install && \
    ldconfig

# Clean up build files to reduce Docker layer size
RUN rm -rf /opencv_build

################################################################################
# Stage 2: Project Builder - Compile C/C++ pose estimation code
################################################################################
FROM --platform=$PLATFORM debian:bullseye-slim AS project-builder

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

# Copy OpenCV installation from Stage 1
COPY --from=opencv-builder /usr/local /usr/local

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy project source code
WORKDIR /build
COPY src/ ./src/
COPY include/ ./include/
COPY CMakeLists.txt ./

# ============================================================================
# Build Project with Minimal Platform-Specific Flags
# ============================================================================
# CMAKE_BUILD_TYPE=Release: Enables -O2 optimization by default
# Additional flags are minimal and match the OpenCV build
# ============================================================================
RUN case "$TARGETPLATFORM" in \
        "linux/amd64") \
            # WSL & Ubuntu: Basic optimizations
            # -O2: Standard optimization level (good balance)
            # No advanced flags needed for MVP
            BUILD_JOBS=4; \
            ;; \
        "linux/arm64") \
            # Raspberry Pi 4: Basic optimizations
            BUILD_JOBS=2; \
            ;; \
        *) \
            BUILD_JOBS=2; \
            ;; \
    esac && \
    mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release .. && \
    make -j${BUILD_JOBS} && \
    make install

################################################################################
# Stage 3: Runtime - Minimal image with only what's needed to run
################################################################################
FROM --platform=$PLATFORM debian:bullseye-slim

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

# Add metadata labels
LABEL maintainer="6-DoF Pose Estimation University Project"
LABEL description="WSL, Ubuntu (amd64) and Raspberry Pi 4 (arm64) support"
LABEL target.platform="${TARGETPLATFORM}"

# ============================================================================
# Install Runtime Dependencies Only
# ============================================================================
# Only install libraries needed to RUN the program (not build it)
# libjpeg62-turbo: Read/write JPEG images
# libpng16-16: Read/write PNG images
# python3: Python interpreter for interactive mode
# python3-pip: Python package manager
# build-essential: GCC/G++ compiler for interactive development
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy OpenCV libraries from Stage 1
COPY --from=opencv-builder /usr/local/lib /usr/local/lib
COPY --from=opencv-builder /usr/local/include/opencv4 /usr/local/include/opencv4

# Copy compiled executable from Stage 2
COPY --from=project-builder /usr/local/bin/pose_estimator /usr/local/bin/

# Update dynamic library cache
RUN ldconfig

# Set working directory where images will be mounted
WORKDIR /data

# ============================================================================
# Container Execution Configuration
# ============================================================================
# When you run: docker run pose-estimator img1.jpg img2.jpg
# It executes: /usr/local/bin/pose_estimator img1.jpg img2.jpg
# ============================================================================
ENTRYPOINT ["/usr/local/bin/pose_estimator"]
CMD ["image1.jpg", "image2.jpg"]

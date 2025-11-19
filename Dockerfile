# Multi-Architecture Dockerfile for 6-DoF Relative Pose Estimation
# Platforms: linux/amd64 (Windows PC/Linux x86_64), linux/arm64 (Raspberry Pi 4)
#
# Build with: make docker-build PLATFORM=<amd64|arm64|multi>
# Or directly: docker buildx build --platform linux/amd64,linux/arm64 -t pose-estimator .

ARG PLATFORM=linux/amd64
ARG OPENCV_VERSION=4.8.1

################################################################################
# Stage 1: OpenCV Builder - Platform-specific optimizations
################################################################################
FROM --platform=$PLATFORM debian:bullseye-slim AS opencv-builder

ARG TARGETPLATFORM
ARG OPENCV_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    ca-certificates \
    pkg-config \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libatlas-base-dev \
    gfortran \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Download OpenCV source
WORKDIR /opencv_build
RUN wget -q -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip -q opencv.zip && \
    rm opencv.zip && \
    mkdir -p opencv-${OPENCV_VERSION}/build

WORKDIR /opencv_build/opencv-${OPENCV_VERSION}/build

# Configure platform-specific build settings
RUN case "$TARGETPLATFORM" in \
        "linux/amd64") \
            # x86_64 optimizations for Windows PC/Linux desktop
            CMAKE_FLAGS=" \
                -D CPU_BASELINE=SSE,SSE2,SSE3 \
                -D CPU_DISPATCH=SSE4_1,SSE4_2,AVX,AVX2 \
                -D ENABLE_AVX=ON \
                -D ENABLE_AVX2=ON \
                -D BUILD_JOBS=8"; \
            ;; \
        "linux/arm64") \
            # ARM64 optimizations for Raspberry Pi 4
            CMAKE_FLAGS=" \
                -D ENABLE_NEON=ON \
                -D ENABLE_VFPV3=ON \
                -D CPU_BASELINE=NEON \
                -D BUILD_JOBS=4"; \
            ;; \
        *) \
            CMAKE_FLAGS="-D BUILD_JOBS=2"; \
            ;; \
    esac && \
    echo "$CMAKE_FLAGS" > /tmp/cmake_flags.txt

# Build OpenCV with platform optimizations
RUN CMAKE_FLAGS=$(cat /tmp/cmake_flags.txt) && \
    cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    # Only build essential modules for pose estimation
    -D BUILD_LIST=core,imgproc,imgcodecs,features2d,calib3d,flann \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_apps=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_opencv_java=OFF \
    # Disable GUI components
    -D WITH_GTK=OFF \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=OFF \
    -D WITH_GSTREAMER=OFF \
    # Enable performance libraries
    -D WITH_TBB=ON \
    -D WITH_OPENMP=ON \
    -D WITH_PTHREADS_PF=ON \
    # Disable Intel-specific optimizations on ARM
    -D WITH_IPP=OFF \
    -D WITH_ITT=OFF \
    # Camera and video support
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_FFMPEG=ON \
    # Enable ORB features (required for this project)
    -D OPENCV_ENABLE_NONFREE=ON \
    # Size optimizations
    -D BUILD_SHARED_LIBS=ON \
    -D OPENCV_SKIP_PYTHON_LOADER=ON \
    $CMAKE_FLAGS \
    ..

# Build with appropriate parallelism
RUN BUILD_JOBS=$(grep "BUILD_JOBS" /tmp/cmake_flags.txt | sed 's/.*BUILD_JOBS=\([0-9]*\).*/\1/') && \
    echo "Building OpenCV for $TARGETPLATFORM with -j${BUILD_JOBS}" && \
    make -j${BUILD_JOBS} && \
    make install && \
    ldconfig

# Clean up to reduce layer size
RUN rm -rf /opencv_build

################################################################################
# Stage 2: Project Builder - Compile pose estimation code
################################################################################
FROM --platform=$PLATFORM debian:bullseye-slim AS project-builder

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

# Copy OpenCV installation from builder stage
COPY --from=opencv-builder /usr/local /usr/local

# Install minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set up build directory
WORKDIR /build

# Copy project sources
COPY src/ ./src/
COPY include/ ./include/
COPY CMakeLists.txt ./

# Build project with platform-specific optimizations
RUN case "$TARGETPLATFORM" in \
        "linux/amd64") \
            # x86_64 optimizations
            CXXFLAGS="-O3 -march=x86-64 -mtune=generic -msse2 -msse3"; \
            BUILD_JOBS=8; \
            ;; \
        "linux/arm64") \
            # ARM64/Raspberry Pi 4 optimizations
            CXXFLAGS="-O3 -march=armv8-a+crc -mtune=cortex-a72 -mfpu=neon-fp-armv8"; \
            BUILD_JOBS=4; \
            ;; \
        *) \
            CXXFLAGS="-O2"; \
            BUILD_JOBS=2; \
            ;; \
    esac && \
    mkdir -p build && cd build && \
    cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_CXX_FLAGS="$CXXFLAGS" \
        .. && \
    make -j${BUILD_JOBS} && \
    make install

################################################################################
# Stage 3: Runtime - Minimal production image
################################################################################
FROM --platform=$PLATFORM debian:bullseye-slim

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

# Add metadata
LABEL maintainer="6-DoF Pose Estimation"
LABEL description="Multi-arch: Windows PC (amd64) and Raspberry Pi 4 (arm64)"
LABEL target.platform="${TARGETPLATFORM}"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    libgomp1 \
    libatlas3-base \
    libtbb2 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libv4l-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy OpenCV libraries
COPY --from=opencv-builder /usr/local/lib /usr/local/lib
COPY --from=opencv-builder /usr/local/include/opencv4 /usr/local/include/opencv4

# Copy compiled executable
COPY --from=project-builder /usr/local/bin/pose_estimator /usr/local/bin/

# Update library cache
RUN ldconfig

# Display build information
RUN echo "Built for platform: $TARGETPLATFORM" > /etc/build-info

# Set working directory for input data
WORKDIR /data

# Configure entrypoint
ENTRYPOINT ["/usr/local/bin/pose_estimator"]
CMD ["image1.jpg", "image2.jpg"]

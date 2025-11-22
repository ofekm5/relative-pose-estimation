cmake_minimum_required(VERSION 3.10)
project(RelativePoseEstimation)

# ============================================================================
# C++ Standard Configuration
# ============================================================================
# Require C++11 for modern features (auto, nullptr, range-based for, etc.)
# ============================================================================
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Platform Detection & Minimal Optimization Flags
# ============================================================================
# CMAKE_SYSTEM_PROCESSOR: Auto-detected by CMake (x86_64, aarch64, etc.)
# CMAKE_BUILD_TYPE=Release: Automatically enables -O2 optimization
#
# For university MVP, we use minimal flags:
#   - WSL/Ubuntu (x86_64): Let compiler choose best flags automatically
#   - Raspberry Pi 4 (ARM64): Enable NEON if available
# ============================================================================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
    # ARM64 platform detected (Raspberry Pi 4)
    message(STATUS "ARM64 platform detected - enabling NEON optimizations")
    # Check if compiler supports -mfpu=neon flag
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
    if(COMPILER_SUPPORTS_NEON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)")
    # x86_64 platform detected (WSL, Ubuntu)
    message(STATUS "x86_64 platform detected - using default optimizations")
    # No additional flags needed; -O2 from Release mode is sufficient
else()
    # Unknown platform
    message(STATUS "Unknown platform: ${CMAKE_SYSTEM_PROCESSOR} - using default optimizations")
endif()

# ============================================================================
# Find Dependencies
# ============================================================================
# OpenCV: Computer vision library (ORB, Essential Matrix, RecoverPose)
# ============================================================================
find_package(OpenCV REQUIRED)

# ============================================================================
# Include Directories
# ============================================================================
# Tell compiler where to find header files
# ============================================================================
include_directories(
    ${OpenCV_INCLUDE_DIRS}           # OpenCV headers
    ${CMAKE_SOURCE_DIR}/include      # Project headers
    ${CMAKE_SOURCE_DIR}/src          # Source-level headers
)

# ============================================================================
# Source Files
# ============================================================================
# Automatically collect all .c and .cpp files in src/ directory
# This allows partners to add files without modifying CMakeLists.txt
# ============================================================================
file(GLOB SOURCES
    "src/*.c"
    "src/*.cpp"
)

# ============================================================================
# Build Executable
# ============================================================================
# Create the pose_estimator binary from all source files
# ============================================================================
add_executable(pose_estimator ${SOURCES})

# ============================================================================
# Link Libraries
# ============================================================================
# Link OpenCV libraries to the executable
# ============================================================================
target_link_libraries(pose_estimator ${OpenCV_LIBS})

# ============================================================================
# Installation Target
# ============================================================================
# 'make install' will copy the binary to /usr/local/bin (in Docker)
# ============================================================================
install(TARGETS pose_estimator DESTINATION bin)

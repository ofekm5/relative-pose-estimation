cmake_minimum_required(VERSION 3.10)
project(RelativePoseEstimation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Source files
file(GLOB SOURCES
    "src/*.c"
    "src/*.cpp"
)

# Create executable
add_executable(pose_estimator ${SOURCES})

# Link OpenCV libraries
target_link_libraries(pose_estimator ${OpenCV_LIBS})

# Install target
install(TARGETS pose_estimator DESTINATION bin)

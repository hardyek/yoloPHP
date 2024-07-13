cmake_minimum_required(VERSION 3.10)
project(yolov8_cpp)

# C++17 Required for PyTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set Torch_DIR to find LibTorch
set(Torch_DIR /home/hardy/projects/yoloPHP/libtorch/share/cmake/Torch)
list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")

# Set OpenCV_DIR
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
list(APPEND CMAKE_PREFIX_PATH "${OpenCV_DIR}")

# CUDA Compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)

# Find packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add library
add_library(yolov8 SHARED yolov8.cpp)

# Link libraries
target_link_libraries(yolov8
    ${OpenCV_LIBS}
    "${TORCH_LIBRARIES}"
)

# Ensure correct C++ standard is used
set_property(TARGET yolov8 PROPERTY CXX_STANDARD 17)

# Verbose output for build
set(CMAKE_VERBOSE_MAKEFILE ON)
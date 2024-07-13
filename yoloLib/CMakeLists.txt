cmake_minimum_required(VERSION 3.10)
project(yoloLib)

# C++17 Required for PyTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)

# Set Torch_DIR to find LibTorch
set(Torch_DIR /home/hardy/projects/yoloPHP/yoloLib/libtorch/share/cmake/Torch)
list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")

# Find packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${Torch_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

# Add the preprocessing library
find_library(PREPROCESSING_LIB NAMES preprocessingLib PATHS /usr/local/lib)

# Add library
add_library(yoloLib SHARED src/yolov8.cpp include/yolov8.h)

# Link libraries
target_link_libraries(yoloLib "${TORCH_LIBRARIES}" ${PREPROCESSING_LIB})

# Ensure correct C++ standard is used
set_property(TARGET yoloLib PROPERTY CXX_STANDARD 17)
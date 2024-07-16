cmake_minimum_required(VERSION 3.10)
project(YOLO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc) #FILEPATH (should have been installed here but may not have been.)

# Set Torch_DIR to find LibTorch
set(Torch_DIR /home/hardy/projects/yoloPHP/libtorch/share/cmake/Torch) #FILEPATH
list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")

# Find packages
find_package(Torch REQUIRED)

# Include directories
include_directories(${Torch_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

# Add the path to stb headers
include_directories(${CMAKE_SOURCE_DIR}/include/stb)

# Add library
add_library(YOLO SHARED src/yolov8.cpp src/stb_image_impl.cpp include/yolov8.h)

# Link libraries
target_link_libraries(YOLO "${TORCH_LIBRARIES}")

# Ensure correct C++ standard is used
set_property(TARGET YOLO PROPERTY CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.10)
project(yoloLib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)

set(Torch_DIR /home/hardy/projects/yoloPHP/yoloLib/libtorch/share/cmake/Torch)
list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")

set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
list(APPEND CMAKE_PREFIX_PATH "${OpenCV_DIR}")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Add preprocessingLib
add_subdirectory(${CMAKE_SOURCE_DIR}/../preprocessingLib ${CMAKE_BINARY_DIR}/preprocessingLib)

include_directories(${OpenCV_INCLUDE_DIRS} ${Torch_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/../preprocessingLib/include)

add_library(yoloLib SHARED src/yolov8.cpp)

target_link_libraries(yoloLib PRIVATE "${TORCH_LIBRARIES}" preprocessingLib ${OpenCV_LIBS})

set_property(TARGET yoloLib PROPERTY CXX_STANDARD 17)
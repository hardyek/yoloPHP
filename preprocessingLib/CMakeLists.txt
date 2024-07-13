cmake_minimum_required(VERSION 3.10)
project(preprocessingLib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(preprocessingLib STATIC src/preprocessing.cpp)

target_link_libraries(preprocessingLib ${OpenCV_LIBS})
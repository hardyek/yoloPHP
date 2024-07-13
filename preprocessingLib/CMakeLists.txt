cmake_minimum_required(VERSION 3.10)
project(preprocessingLib)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(preprocessingLib SHARED src/preprocessing.cpp)
target_link_libraries(preprocessingLib ${OpenCV_LIBS})

include_directories(include)
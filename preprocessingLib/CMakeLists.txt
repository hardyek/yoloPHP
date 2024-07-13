cmake_minimum_required(VERSION 3.10)
project(preprocessingLib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

add_library(preprocessingLib SHARED src/preprocessing.cpp)

target_link_libraries(preprocessingLib ${OpenCV_LIBS})

install(TARGETS preprocessingLib DESTINATION /usr/local/lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION /usr/local/include)
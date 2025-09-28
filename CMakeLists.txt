cmake_minimum_required(VERSION 3.10)
project(SimpleCVProject CXX)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_FOUND}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(app main.cpp)

target_link_libraries(app ${OpenCV_LIBS})
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(render_trajectory LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(RTRenderer REQUIRED)

add_executable(render_trajectory main.cpp)

target_link_libraries(render_trajectory PRIVATE RTRenderer::RTRenderer)

install(TARGETS render_trajectory
    RUNTIME DESTINATION bin
)

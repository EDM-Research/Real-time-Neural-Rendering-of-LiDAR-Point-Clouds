cmake_minimum_required(VERSION 3.12)
project(PCL_renderer_trajectory)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE PATH "Path to nvcc")

add_subdirectory(render_trajectory)

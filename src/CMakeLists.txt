cmake_minimum_required(VERSION 3.12)
project(PCL_Renderer)

set(Torch_DIR ~/lib/libtorch/share/cmake/Torch CACHE PATH "Path to TorchConfig.cmake")
set(Torch_TensorRT_DIR ~/lib/torch_tensorrt CACHE PATH "Path to torch_tensorrt folder")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE PATH "Path to nvcc")

add_subdirectory(RTRenderer)
add_subdirectory(Synthetic)

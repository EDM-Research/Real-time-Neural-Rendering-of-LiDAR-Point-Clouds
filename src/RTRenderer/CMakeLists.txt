cmake_minimum_required(VERSION 3.15)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(RTRenderer LANGUAGES CXX CUDA)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(E57Format REQUIRED)
find_package(glm REQUIRED)

option(WITH_TENSOR_RT "Enable Torch-TensorRT support" ON)

if(WITH_TENSOR_RT)
    find_library(TORCH_TENSORRT_RUNTIME_LIB torchtrt_runtime)

    if(TORCH_TENSORRT_RUNTIME_LIB)
        find_path(Torch_TensorRT_BASE_DIR
            NAMES torch_tensorrt/torch_tensorrt.h
            PATHS /usr/local/include /usr/include
        )

        if (Torch_TensorRT_BASE_DIR)
            set(Torch_TensorRT_INCLUDE_DIR "${Torch_TensorRT_BASE_DIR}/torch_tensorrt")
            message(STATUS "Found Torch-TensorRT include dir: ${Torch_TensorRT_INCLUDE_DIR}")
            include_directories(${Torch_TensorRT_INCLUDE_DIR})
        else()
            message(FATAL_ERROR "Torch-TensorRT headers not found")
        endif()

        include_directories(${Torch_TensorRT_DIR}/include)
        set(TORCH_TENSORRT_LIBRARIES
            ${TORCH_TENSORRT_RUNTIME_LIB}
        )
    message(STATUS "Found Torch_TensorRT libraries: " ${TORCH_TENSORRT_LIBRARIES})
    else()
        message(FATAL_ERROR "Torch-TensorRT libraries not found")
    endif()
endif()

add_library(RTRenderer SHARED)

target_compile_options(RTRenderer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 20012 -rdc=true -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -g>
)


set_target_properties(RTRenderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(RTRenderer PROPERTIES CUDA_ARCHITECTURES "89")

file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*")
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*")

target_sources(RTRenderer PRIVATE ${sources})
target_include_directories(RTRenderer PRIVATE include)
target_include_directories(RTRenderer PUBLIC
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    )

if(WITH_TENSOR_RT)
    target_include_directories(RTRenderer
        PUBLIC
            ${Torch_TensorRT_INCLUDE_DIR}
        )
endif()

target_link_libraries(RTRenderer PUBLIC
    glm::glm
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES} "-Wl,--no-as-needed" ${TORCH_TENSORRT_LIBRARIES} "-Wl,--as-needed"
    E57Format
)

set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)

install(FILES ${headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RTRenderer
)

install(TARGETS RTRenderer
  EXPORT RTRendererTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RTRenderer
)

install(EXPORT RTRendererTargets
  FILE RTRendererTargets.cmake
  NAMESPACE RTRenderer::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RTRenderer
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/RTRendererConfigVersion.cmake
  VERSION 1.0.0
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RTRendererConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/RTRendererConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RTRenderer
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/RTRendererConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/RTRendererConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RTRenderer
)

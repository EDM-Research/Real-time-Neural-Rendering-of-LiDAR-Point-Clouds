cmake_minimum_required(VERSION 3.5)

project(RTRenderer)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(E57Format REQUIRED)

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

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")

set(SOURCES 
    project_cloud.cu project_cloud.h
    CameraCalibration.cpp CameraCalibration.h
    Config.h
    Frustum.cpp Frustum.h
    Octreegrid.h
    PointCloudReader.cpp PointCloudReader.h
    Utils.cpp Utils.h
)

cuda_add_library(ProjectCloudCUDA SHARED project_cloud.cu)
target_include_directories(ProjectCloudCUDA
    PUBLIC
        ${TORCH_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

if(WITH_TENSOR_RT)
    target_include_directories(ProjectCloudCUDA
        PUBLIC
        ${Torch_TensorRT_INCLUDE_DIR}
        )
endif()


add_library(RTRenderer SHARED ${SOURCES}
    cloudreader.h cloudreader.cpp
    tinyply.h)

target_include_directories(RTRenderer
    PUBLIC
        ${TORCH_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

if(WITH_TENSOR_RT)
    target_include_directories(RTRenderer
        PUBLIC
            ${Torch_TensorRT_INCLUDE_DIR}
        )
endif()

target_link_libraries(RTRenderer
    PUBLIC
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES} "-Wl,--no-as-needed" ${TORCH_TENSORRT_LIBRARIES} "-Wl,--as-needed"
        ProjectCloudCUDA
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_cudart_LIBRARY}
        E57Format
)

set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)

install(FILES project_cloud.h
              CameraCalibration.h
              Octreegrid.h
              cloudreader.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RTRenderer
)

install(TARGETS RTRenderer ProjectCloudCUDA
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

cmake_minimum_required(VERSION 3.5)

project(SyntheticGeneration)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(synthetic_generation SyntheticGeneration.cpp
Utils.cpp Utils.h)
target_link_libraries(synthetic_generation  PRIVATE RTRenderer )

install(TARGETS synthetic_generation
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )


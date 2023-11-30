# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# static library
add_library(pano STATIC)
target_sources(pano PRIVATE stitcher.cpp common.cpp)

# link opencv
target_link_libraries(pano PRIVATE argparse)
target_link_libraries(pano PUBLIC ${OpenCV_LIBS})
target_link_libraries(pano PUBLIC MPI::MPI_CXX)

target_include_directories(pano PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# TODO: cuda, mpi, openmp
add_subdirectory(seq)
add_subdirectory(opencv)
add_subdirectory(mpi)
add_subdirectory(openmp)
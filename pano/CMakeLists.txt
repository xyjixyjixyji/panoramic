# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP REQUIRED)
include_directories(${OpenMP_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories("/usr/local/cuda-11.7/lib64/")

# static library
add_library(pano STATIC stitcher.cpp common.cpp)

# link opencv, MPI, OpenMP, and CUDA
target_link_libraries(pano PRIVATE argparse)
target_link_libraries(pano PUBLIC ${OpenCV_LIBS})
target_link_libraries(pano PUBLIC MPI::MPI_CXX)
target_link_libraries(pano PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(pano PUBLIC ${CUDA_LIBRARIES} ${CUDA_cudart_LIBRARY})
target_include_directories(pano PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# TODO: cuda, mpi, openmp
add_subdirectory(cuda)
add_subdirectory(seq)
add_subdirectory(opencv)
add_subdirectory(mpi)
add_subdirectory(openmp)
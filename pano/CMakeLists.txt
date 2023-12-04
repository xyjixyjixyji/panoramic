# static library
add_library(pano STATIC stitcher.cpp common.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wpedantic -Werror")

# include
target_include_directories(pano PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# argparse
target_link_libraries(pano PRIVATE argparse)

enable_language(CUDA)

# cuda, mpi, openmp, opencv
add_subdirectory(cuda)
add_subdirectory(seq)
add_subdirectory(opencv)
add_subdirectory(mpi)
add_subdirectory(openmp)
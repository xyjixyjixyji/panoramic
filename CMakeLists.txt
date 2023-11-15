cmake_minimum_required(VERSION 3.14)

project(pano)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wpedantic -Werror")

add_subdirectory(pano)

add_executable(pano_cmd main.cpp)
target_link_libraries(pano_cmd pano)

# find_package(OpenMP)
# if(OpenMP_FOUND)
#     add_subdirectory(pano/openmp) # the openmp itself links lopenmp, we are not doing it here
# endif()

# find_package(MPI)
# if(MPI_FOUND)
#     add_subdirectory(pano/mpi)
# endif()

# TODO: cuda


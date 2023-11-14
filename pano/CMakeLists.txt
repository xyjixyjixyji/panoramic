# static library
add_library(libpano STATIC)
target_sources(libpano PRIVATE pano.cpp)

target_include_directories(libpano PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(seq)
# TODO: cuda, mpi, openmp

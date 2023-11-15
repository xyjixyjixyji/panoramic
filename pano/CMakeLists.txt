# OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# static library
add_library(libpano STATIC)
target_sources(libpano PRIVATE pano.cpp)

# link opencv
target_link_libraries(libpano ${OpenCV_LIBS})

target_include_directories(libpano PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# TODO: cuda, mpi, openmp
add_subdirectory(seq)

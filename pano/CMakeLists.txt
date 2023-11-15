# OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# static library
add_library(pano STATIC)
target_sources(pano PRIVATE pano.cpp)

# link opencv
target_link_libraries(pano ${OpenCV_LIBS})

target_include_directories(pano PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# TODO: cuda, mpi, openmp
add_subdirectory(seq)

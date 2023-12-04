project(pano_cuda)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories("/usr/local/cuda-11.7/lib64/")

file(GLOB CUDA_SOURCES "*.cu")

cuda_add_executable(pano_cuda ${CUDA_SOURCES})

target_link_libraries(pano_cuda pano ${CUDA_LIBRARIES} ${CUDA_cudart_LIBRARY})
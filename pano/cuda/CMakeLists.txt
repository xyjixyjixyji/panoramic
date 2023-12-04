find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(pano PUBLIC ${CUDA_LIBRARIES})
set_target_properties(pano PROPERTIES CUDA_ARCHITECTURES OFF)

target_sources(pano PRIVATE cudaHarrisDet.cu cudaHarrisMatch.cu cudaRansac.cu)

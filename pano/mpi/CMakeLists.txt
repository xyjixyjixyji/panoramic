find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(pano PUBLIC MPI::MPI_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 ${MPI_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

target_sources(pano PRIVATE mpiHarrisDet.cpp mpiHarrisMatch.cpp mpiRansac.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 ${MPI_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

target_link_libraries(pano PRIVATE MPI::MPI_CXX)
target_sources(pano PRIVATE mpiHarrisDet.cpp mpiHarrisMatch.cpp)

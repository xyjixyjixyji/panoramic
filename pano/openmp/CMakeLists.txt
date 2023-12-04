find_package(OpenMP REQUIRED)
include_directories(${OpenMP_INCLUDE_DIRS})
target_link_libraries(pano PUBLIC OpenMP::OpenMP_CXX)

target_sources(pano PRIVATE ompHarrisDet.cpp ompHarrisMatch.cpp ompRansac.cpp)
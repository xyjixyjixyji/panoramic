# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(pano PUBLIC ${OpenCV_LIBS})

target_sources(pano PRIVATE ocvRansac.cpp ocvHarrisDet.cpp ocvHarrisMatch.cpp)

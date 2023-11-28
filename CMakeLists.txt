cmake_minimum_required(VERSION 3.14)

project(pano)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wpedantic -Werror")


# third party
include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

add_subdirectory(pano)

add_executable(pano_cmd main.cpp)
target_link_libraries(pano_cmd pano argparse)

# set_target_properties(pano_cmd PROPERTIES COMPILE_FLAGS "-g")


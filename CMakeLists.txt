cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall" "-pedantic")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(yolov3_object_detector)

find_package(OpenCV 4.1 REQUIRED)
find_package(Boost 1.60.0 COMPONENTS filesystem)

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable (yolov3_object_detector src/yolov3_object_detector.cpp)
target_link_libraries (yolov3_object_detector ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})


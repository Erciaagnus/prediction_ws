cmake_minimum_required(VERSION 3.0.2)
project(onedayalg)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  msgs
  roscpp
  rospy
  std_msgs
  tf
  roslib
)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME} 
  CATKIN_DEPENDS geometry_msgs msgs roscpp rospy std_msgs tf roslib
  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME} SHARED
  lib/onedayalg.cpp
)


add_executable(${PROJECT_NAME}_node src/main.cpp)
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


## Specify libraries to link a library or executable target against

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${catkin_LIBRARY_DIRS}
  ${PROJECT_NAME}
	
)

install(DIRECTORY include
DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
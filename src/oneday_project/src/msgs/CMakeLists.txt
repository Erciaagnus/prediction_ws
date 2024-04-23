cmake_minimum_required(VERSION 3.0.2)
project(msgs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
 FILES
 object_array_msg.msg
 object_msg.msg
 map_array_msg.msg
 map_msg.msg
 point_msg.msg
 dataset_msg.msg
 dataset_array_msg.msg
)
 

## Generate added messages and services with any dependencies listed here
generate_messages(
 DEPENDENCIES
 geometry_msgs
 nav_msgs
 sensor_msgs
 std_msgs
 visualization_msgs
)

catkin_package(CATKIN_DEPENDS
 geometry_msgs
 nav_msgs
 sensor_msgs
 std_msgs
 visualization_msgs
  message_runtime
)


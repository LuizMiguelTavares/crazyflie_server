cmake_minimum_required(VERSION 3.0.2)
project(crazyflie_server)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  std_srvs
)

catkin_package(
  CATKIN_DEPENDS std_msgs geometry_msgs std_srvs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install script for directory: /home/apbraga/Courses/Udacity Autonomous Vehicle Engineer Nanodegree/09 Capstone Project/ros/src/waypoint_loader

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/apbraga/Courses/Udacity Autonomous Vehicle Engineer Nanodegree/09 Capstone Project/ros/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/apbraga/Courses/Udacity Autonomous Vehicle Engineer Nanodegree/09 Capstone Project/ros/build/waypoint_loader/catkin_generated/installspace/waypoint_loader.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/waypoint_loader/cmake" TYPE FILE FILES
    "/home/apbraga/Courses/Udacity Autonomous Vehicle Engineer Nanodegree/09 Capstone Project/ros/build/waypoint_loader/catkin_generated/installspace/waypoint_loaderConfig.cmake"
    "/home/apbraga/Courses/Udacity Autonomous Vehicle Engineer Nanodegree/09 Capstone Project/ros/build/waypoint_loader/catkin_generated/installspace/waypoint_loaderConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/waypoint_loader" TYPE FILE FILES "/home/apbraga/Courses/Udacity Autonomous Vehicle Engineer Nanodegree/09 Capstone Project/ros/src/waypoint_loader/package.xml")
endif()


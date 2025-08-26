cmake_minimum_required(VERSION 3.21)
project(SpacePerception LANGUAGES CXX)

add_library(SpacePerception SHARED
    Perception/dllmain.cpp
    Perception/pch.cpp
)

target_include_directories(SpacePerception PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Perception
    ${CMAKE_SOURCE_DIR}/union-api/include
    ${CMAKE_SOURCE_DIR}/gothic-api/include
)

target_link_libraries(SpacePerception
    union-api
    gothic-api
)

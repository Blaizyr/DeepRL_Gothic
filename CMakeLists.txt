cmake_minimum_required(VERSION 3.21)
project(DeepRL_Gothic VERSION 1.0.0 LANGUAGES CXX)

 set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

 if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()


 add_subdirectory(union-api)
add_subdirectory(gothic-api)


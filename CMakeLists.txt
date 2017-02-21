cmake_minimum_required(VERSION 3.5)
project(cascade_generator)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp
        )

find_package( OpenCV 3.1 REQUIRED )


add_executable(cascade_generator ${SOURCE_FILES})

target_link_libraries( cascade_generator ${OpenCV_LIBS} )
cmake_minimum_required(VERSION 3.14)
project(Doom_test)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
)

add_executable(Doom_test main.cpp)
TARGET_LINK_LIBRARIES(Doom_test dl)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        )

include_directories("/home/ruilong/CLionProjects/Doom_test/include")
target_link_libraries(Doom_test "/home/ruilong/CLionProjects/Doom_test/lib/libvizdoom.so")
target_link_libraries(Doom_test "/home/ruilong/CLionProjects/Doom_test/lib/libORB_SLAM2.so")
cmake_minimum_required(VERSION 3.29)
project(NeuralHMM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/external)

set(SOURCES
        src/main.cpp
        src/model.h
        src/model.cpp
)

add_executable(NeuralHMM ${SOURCES})
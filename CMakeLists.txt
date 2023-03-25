cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

find_package(Torch REQUIRED)

add_executable(currentExecutable  main.cpp dcgan.cpp)
target_link_libraries(currentExecutable "${TORCH_LIBRARIES}")
set_property(TARGET currentExecutable PROPERTY CXX_STANDARD 14)
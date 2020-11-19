cmake_minimum_required (VERSION 3.16.3)
project (fft LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "nvcc")

add_executable(2dfft main.cpp 2dfft.cu 2dfft.cpp)
target_link_libraries(2dfft fftw3)
target_link_libraries(2dfft cufft)
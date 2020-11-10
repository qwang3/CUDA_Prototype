cmake_minimum_required (VERSION 3.16.3)
project (fft LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "nvcc")

add_executable(1dfft main.cpp 1dfft.cu 1dfft.cpp)
target_link_libraries(1dfft fftw3)
target_link_libraries(1dfft cufft)
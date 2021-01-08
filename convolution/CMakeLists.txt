cmake_minimum_required (VERSION 3.16.3)
project (fft LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

# OPEN_MP settings 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")

add_executable(conv1 main.cpp conv_fftw.cpp conv_cufft.cu)
target_link_libraries(conv1 fftw3)
target_link_libraries(conv1 fftw3_omp)
target_link_libraries(conv1 cufft)
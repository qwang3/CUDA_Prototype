cmake_minimum_required (VERSION 3.16.3)
project (fft LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -G")

include_directories(/usr/local/cuda/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")


add_executable(conv main.cpp conv_fftw.cpp conv_cufft.cu conv_cufft2.cu)
target_link_libraries(conv fftw3f)
target_link_libraries(conv fftw3)
target_link_libraries(conv cufft)

add_executable(plot compare.cpp conv_fftw.cpp conv_cufft.cu conv_cufft2.cu)
target_link_libraries(plot fftw3f)
target_link_libraries(conv fftw3)
target_link_libraries(plot cufft)

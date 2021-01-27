#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

typedef float2 Complex;

void map_memory(float2 array[], int size);

float convolve_fftw(Complex input_array_1[], Complex input_array_2[], 
            Complex output_array[], int N_max, int batch);

float convolve_cufft_1(Complex input_array_1[], Complex input_array_2[], 
            Complex output_array[], int N_max, int batch);

float convolve_cufft_2(Complex input_array_1[], Complex input_array_2[], 
            Complex output_array[], int N_max, int batch);

float convolve_cufft_3(Complex input_array_1[], Complex input_array_2[], 
            Complex output_array[], int N_max, int batch);

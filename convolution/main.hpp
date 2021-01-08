#include <omp.h>
#include <cmath>
#include <cstdio>

double convolve_fftw(float input_array_1[], float input_array_2[], 
            float output_array[], int N_max, int batch);

double convolve_cufft(float input_array_1[], float input_array_2[], 
            float output_array[], int N_max, int batch);
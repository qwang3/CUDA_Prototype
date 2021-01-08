#include <omp.h>
#include <cmath>
#include <cstdio>

double convolve_fftw(double input_array_1[], double input_array_2[], 
            double output_array[], int N_max, int batch);

double convolve_cufft(double input_array_1[], double input_array_2[], 
            double output_array[], int N_max, int batch);
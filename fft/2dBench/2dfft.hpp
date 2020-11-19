#include <cuda_runtime.h>
#include <cufft.h>

#include <fftw3.h>

#include <ctime>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cmath>

__global__ void real2complex(double *f, cufftDoubleComplex *fc, int N);
__global__ void complex2real(cufftDoubleComplex *fc, double *f, int N);

float fft_cuda(double** idata, double** odata, int Nx, int Ny);
float fft_fftw2d(double** idata, double** odata, int Nx, int Ny);
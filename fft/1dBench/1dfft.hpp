#include <cuda_runtime.h>
#include <cufft.h>

#include <fftw3.h>

#include <ctime>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

__global__ void real2complex(double *f, cufftDoubleComplex *fc, int N);
__global__ void complex2real(cufftDoubleComplex *fc, double *f, int N);

std::chrono::microseconds fft_cuda(const double* idata, double* odata, int Nx);
std::chrono::microseconds fft_fftw(const double* idata, double* odata, int Nx);
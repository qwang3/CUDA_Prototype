#include <ctime>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

#include <fftw3.h>

#include <cuda_runtime.h>
#include <cufft.h>

using namespace std; 

__global__ void real2complex(double *f, cufftComplex *fc, int N);
__global__ void complex2real(cufftComplex *fc, double *f, int N);

void usage(int argc, char **argv) {
    cout << "usage: " << argv[0] << " Nx Ny" << endl;
    cout << "Nx (int): number of elemenst in x direction" << endl;
    cout << "Nx (int): number of elemenst in y direction" << endl;
    cout << "Nz (int): number of elemenst in z direction" << endl;
    cout << "NCPU (int): number of GPUs tp use" << endl;
}
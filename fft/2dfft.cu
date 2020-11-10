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
}

int main(int argc, char **argv) {

    if (argc != 3) {
        usage(argc, argv);
        exit(1);
    }
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);

    /* randomly generate data */
    double *idata = new double[Nx*Ny];
    double *odata = new double[Nx*Ny];
    for (int i=0; i<Nx*Ny; i++) {
        idata[i] = rand() / (double) RAND_MAX;
    }

    /* Allocate memory for data on device, then copy data */
    cout << "Allocating memory for data on device" << endl;
    double *idata_c;
    cufftComplex *idata_cx;
    double *odata_c; 
    cufftComplex *odata_cx;
    cudaMalloc(&odata_c, sizeof(double) * Nx*Ny);
    cudaMalloc(&idata_c, sizeof(double) * Nx*Ny);
    cudaMalloc(&idata_cx, sizeof(cufftComplex) * Nx*Ny);
    cudaMalloc(&odata_cx, sizeof(cufftComplex) * Nx*Ny);

    cudaMemcpy(idata_c, idata, sizeof(double) * Nx*Ny, cudaMemcpyHostToDevice);

    /* Convert data into cufftComplex */
    /* set 1 block with 256 threads */
    cout << "converting real2complex" << endl;
    real2complex<<<2, 128>>>(idata_c, idata_cx, Nx*Ny);

    /* FFT Plans */
    int n[2] = {Nx, Ny};
    cufftHandle plan;
    cufftPlanMany(&plan, 2, n,
                    NULL, 1, 0, 
                    NULL, 1, 0, 
                    CUFFT_C2C, 1);

    /* Forward FFT */
    cufftExecC2C(plan, idata_cx, odata_cx, CUFFT_FORWARD);

    /* Inverse FFT */
    cufftExecC2C(plan, odata_cx, idata_cx, CUFFT_INVERSE);

    /* Convert cufft back to double array */
    /* set 1 block with 256 threads */

    cout << "converting complex2real" << endl;
    complex2real<<<2, 128>>>(idata_cx, odata_c, Nx);

    cudaMemcpy(odata, odata_c, sizeof(double)*Nx, cudaMemcpyDeviceToHost);

    /* Normalize result */
    for (int i=0; i < Nx*Ny; i++) {
        odata[i] /= Nx*Ny;
    }


    for (int i = 0; i < Nx; i++){
        if (abs(idata[i] - odata[i]) > 1e-6) {
            cout << "[ERROR] Mismatch at " << i;
            cout << " idata[" << i << "]= " << idata[i];
            cout << " odata[" << i << "]= " << odata[i] << endl;
        }
    }

    cufftDestroy(plan);
    free(odata);
    free(idata);
    cudaFree(idata_c);
    cudaFree(idata_cx);
    cudaFree(odata_c);
}

/* convert a double array to cuffComplex data type. Imaginary parts are
 * set to 0 
 */
__global__ void real2complex(double *f, cufftComplex *fc, int N) {
    /* Assume 1D grid of 1D blocks */
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (index < N) {
        fc[index].x = f[index];
        fc[index].y = 0;
        index += stride;
    }
}

/* convert a cuffComplex data type to a double array.
 */
 __global__ void complex2real(cufftComplex *fc, double *f, int N) {
    /* Assume 1D grid of 1D blocks */
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (index < N) {
        f[index] = fc[index].x;
        index += stride;
    }
}
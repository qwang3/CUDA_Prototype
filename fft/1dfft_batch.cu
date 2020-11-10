#include <ctime>
#include <chrono>
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
    cout << "usage: " << argv[0] << " Nx batch" << endl;
    cout << "Nx (int): number of element in array" << endl;
    cout << "batch (int): number of arrys to compute" << endl;
    
}

int main(int argc, char **argv) {

    if (argc != 3) {
        usage(argc, argv);
        exit(1);
    }
    int Nx = atoi(argv[1]);
    int batch = atoi(argv[2]);


    /* randomly generate data */
    double *idata = new double[Nx*batch];
    double *odata = new double[Nx*batch];
    for (int i=0; i<Nx*batch; i++) {
        idata[i] = rand() / (double)RAND_MAX;
    }

    /* Allocate memory for data on device, then copy data */
    cout << "Allocating memory for data on device" << endl;
    double *idata_c;
    cufftComplex *idata_cx;
    double *odata_c; 
    cufftComplex *odata_cx;
    cudaMalloc(&odata_c, sizeof(double) * Nx*batch);
    cudaMalloc(&idata_c, sizeof(double) * Nx*batch);
    cudaMalloc(&idata_cx, sizeof(cufftComplex) * Nx*batch);
    cudaMalloc(&odata_cx, sizeof(cufftComplex) * Nx*batch);

    cudaMemcpy(idata_c, idata, sizeof(double) * Nx*batch, cudaMemcpyHostToDevice);

    /* FFT Plans */
    cufftHandle plan;
    cufftPlan1d(&plan, Nx, CUFFT_C2C, batch);

    /* start the time now */
    auto start = chrono::high_resolution_clock::now();

    /* Convert data into cufftComplex */
    /* set 1 block with 128 threads */
    cout << "converting real2complex" << endl;
    real2complex<<<1, 128>>>(idata_c, idata_cx, Nx*batch);
    cudaDeviceSynchronize();

    /* Forward FFT */
    cout << "Forward FFT" << endl;
    cufftExecC2C(plan, idata_cx, odata_cx, CUFFT_FORWARD);

    /* Inverse FFT */
    cout << "Inverse FFT" << endl;
    cufftExecC2C(plan, odata_cx, idata_cx, CUFFT_INVERSE);

    /* Convert cufft back to double array */
    /* set 1 block with 128 threads */

    cout << "converting complex2real" << endl;
    complex2real<<<1, 128>>>(idata_cx, odata_c, Nx*batch);
    cudaDeviceSynchronize();

    cudaMemcpy(odata, odata_c, sizeof(double)*Nx*batch, cudaMemcpyDeviceToHost);

    /* stop the time */
    auto finish = chrono::high_resolution_clock::now();

    for (int i = 0; i < Nx*batch; i++){
        if (abs(idata[i] - odata[i]) > 1e-6) {
            cout << "[ERROR] Mismatch at " << i;
            cout << " idata[" << i << "]= " << idata[i];
            cout << " odata[" << i << "]= " << odata[i] << endl;
        }
    }

    auto duration = chrono::duration_cast<chrono::milliseconds>(finish - start);
    cout << "Calculation ran for " << duration.count() << "ms" << endl;

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
        f[index] = fc[index].x / N;
        index += stride;
    }
}
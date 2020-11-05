#include <ctime>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

#include <fftw3.h>

#include <cuda_runtime.h>
#include <cufft.h>

using namespace std; 

__global__ void real2complex(float *f, cufftComplex *fc, int N);
__global__ void complex2real(cufftComplex *fc, float *f, int N);

void usage(int argc, char **argv) {
    cout << "usage: " << argv[0] << " Nx" << endl;
    cout << "Nx (int): number of element in array" << endl;
}

int main(int argc, char **argv) {

    if (argc != 2) {
        usage(argc, argv);
        exit(1);
    }
    int Nx = atoi(argv[1]);

    /* randomly generate data */
    float *idata = new float[Nx];
    for (int i=0; i<Nx; i++) {
        idata[i] = rand() / (float)RAND_MAX;
        cout << "idata[" << i << "]= " << idata[i] << endl;
    }

    /* Allocate memory for data on device, then copy data */
    float * idata_c;
    cufftComplex *idata_cx;
    cudaMalloc((void **) idata_c, sizeof(float) * Nx);
    cudaMalloc((void **) idata_cx, sizeof(cufftComplex) * Nx);
    cudaMemcpy(idata_c, idata, sizeof(float) * Nx, cudaMemcpyHostToDevice);

    /* Convert data into cufftComplex */
    /* set 1 block with 256 threads */
    real2complex<<<1, 16>>>(idata_c, idata_cx, Nx);

    /* Convert cufft back to float array */
    /* set 1 block with 256 threads */
    float *odata_c; 
    float *odata = new float[Nx];
    cudaMalloc((void **) odata_c, sizeof(float) * Nx);

    complex2real<<<1, 16>>>(idata_cx, odata_c, Nx);

    cudaMemcpy(odata, odata_c, sizeof(float)*Nx, cudaMemcpyDeviceToHost);

    for (int i = 0; i < Nx; i++){
        if (idata[i] != odata[i]) {
            cout << "[ERROR] Mismatch at " << i;
            cout << " idata[" << i << "]= " << idata[i];
            cout << " odata[" << i << "]= " << odata[i] << endl;
        }
    }
    free(odata);
    free(idata);
    cudaFree(idata_c);
    cudaFree(idata_cx);
    cudaFree(odata_c);
}

/* convert a float array to cuffComplex data type. Imaginary parts are
 * set to 0 
 */
__global__ void real2complex(float *f, cufftComplex *fc, int N) {
    /* Assume 1D grid of 1D blocks */
    printf("This is thread %d: \n", threadIdx.x);
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (index < N) {
        fc[index].x = f[index];
        fc[index].y = 0;
        printf("Thread %d: %d \n", index, f[index]);
        index += stride;
    }
}

/* convert a cuffComplex data type to a float array.
 */
 __global__ void complex2real(cufftComplex *fc, float *f, int N) {
    /* Assume 1D grid of 1D blocks */
    printf("This is thread %d: \n", threadIdx.x);
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (index < N) {
        f[index] = fc[index].x;
        printf("Thread %d: %d \n", index, fc[index].x);
        index += stride;
    }
}
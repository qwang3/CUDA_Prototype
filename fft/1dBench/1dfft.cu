#include "1dfft.hpp"

using namespace std;

/* Calcuate FFT with cuFTT */

float fft_cuda(const double* idata, double* odata, int Nx) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    /* Allocate memory for data on device, then copy data */
    double *idata_c, *odata_c;
    cufftDoubleComplex *idata_cx, *odata_cx;
    cudaMalloc(&odata_c, sizeof(double) * Nx);
    cudaMalloc(&idata_c, sizeof(double) * Nx);
    cudaMalloc(&idata_cx, sizeof(cufftDoubleComplex) * Nx);
    cudaMalloc(&odata_cx, sizeof(cufftDoubleComplex) * Nx);

    cudaMemcpy(idata_c, idata, sizeof(double) * Nx, cudaMemcpyHostToDevice);

    /* Convert data into cufftDoubleComplex */
    /* set 1 block with 256 threads */
    real2complex<<<1, 8>>>(idata_c, idata_cx, Nx);
    cudaDeviceSynchronize();

    /* FFT Plans */
    cufftHandle plan;
    cufftPlan1d(&plan, Nx, CUFFT_Z2Z, 1);


    // auto start = chrono::high_resolution_clock::now();
    /* Forward FFT */
    cudaEventRecord(start);
    cufftExecZ2Z(plan, idata_cx, odata_cx, CUFFT_FORWARD);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    /* stop the time */
    /* std::chrono::_V2::system_clock::time_point finish */ 
    float duration = 0; // milliseconds
    cudaEventElapsedTime(&duration, start, stop);
    /* Convert cufft back to double array */
    /* set 1 block with 8 threads */

    complex2real<<<1, 8>>>(odata_cx, odata_c, Nx);
    cudaDeviceSynchronize();

    cudaMemcpy(odata, odata_c, sizeof(double)*Nx, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(idata_c);
    cudaFree(idata_cx);
    cudaFree(odata_c);

    return duration;
}

/* convert a double array to cuffComplex data type. Imaginary parts are
 * set to 0 
 */
__global__ void real2complex(double *f, cufftDoubleComplex *fc, int N) {
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
 __global__ void complex2real(cufftDoubleComplex *fc, double *f, int N) {
    /* Assume 1D grid of 1D blocks */
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (index < N) {
        f[index] = fc[index].x;
        index += stride;
    }
}
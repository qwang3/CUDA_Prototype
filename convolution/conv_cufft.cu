#include <cufft.h>

#include "main.hpp"
#include "gpuErrchk.hpp"

static __global__ void point_mul(cufftComplex *in_one, const cufftComplex *in_two,  
                                 int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        cufftComplex in_1 = in_one[idx];
        cufftComplex in_2 = in_two[idx];
        cufftComplex out;

        out.x = (in_1.x * in_2.x - in_1.y * in_2.y)*scale;
        out.y = (in_1.x * in_2.y + in_1.y * in_2.x)*scale;
        in_one[idx] = out;
        // printf("    threadIdx %d: out = %f out = %f \n", idx, in_one[idx].x, in_one[idx].y);

        idx += blockDim.x*gridDim.x;
    }
}

static __global__ void print_mem(cufftComplex * mem, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i=idx; i<N; i+= blockDim.x*gridDim.x) {
        if (mem[i].x != 0) {
            printf("    threadIdx %d: mem[%d].x= %f .y = %f\n", i, i, mem[i].x, mem[i].y);
        }
    }
}

float convolve_cufft_1(Complex input_array_1[], Complex input_array_2[], 
            Complex output_array[], int N_max, int batch) {
    cufftComplex *in_one, *in_two;
    cufftHandle fft_plan;
    int dim[3] = {N_max, N_max, N_max};
    int N_total = pow(N_max, 3);
    // int n_block = 32;
    // int n_thread = 1024;

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    
    /* allocate memory */
    gpuErrchk(cudaMalloc(&in_one, sizeof(cufftComplex)*N_total*batch));
    gpuErrchk(cudaMalloc(&in_two, sizeof(cufftComplex)*N_total*batch));

    int minGridSize, gridSize, blockSize;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, point_mul, 0, N_total*batch));
    gridSize = (N_total*batch + blockSize - 1) / blockSize; 

    cufftPlanMany(&fft_plan, 3, dim, 
        NULL, 1, N_total,  
        NULL, 1, N_total, CUFFT_C2C, batch);

    /* for debugging */


    /* copy over input */
    cudaEventRecord(start);
    gpuErrchk(cudaMemcpy(in_one, input_array_1, sizeof(Complex)*N_total*batch, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(in_two, input_array_2, sizeof(Complex)*N_total*batch, cudaMemcpyHostToDevice));


    /* F = conv(f) */
    cufftExecC2C(fft_plan, in_one, in_one, CUFFT_FORWARD);
    /* G = conv(g) */
    cufftExecC2C(fft_plan, in_two, in_two, CUFFT_FORWARD);

    /* F dot G */
    // cudaDeviceSynchronize();
    point_mul<<<gridSize, blockSize>>>(in_one, in_two, N_total*batch, 1.0f/N_total);
    // cudaDeviceSynchronize();
    /* conv^-1(F dot G) */
    // printf("### [CUDA] Printing out before fft ..\n");
    // print_mem<<<gridSize, blockSize>>>(in_one, N_total*batch);
    // cudaDeviceSynchronize();

    cufftExecC2C(fft_plan, in_one, in_one, CUFFT_INVERSE);
    // cudaDeviceSynchronize();

    // printf("### [CUDA] Printing out after fft..\n");
    // print_mem<<<gridSize, blockSize>>>(out, N_total*batch);

    gpuErrchk(cudaMemcpy(output_array, in_one, sizeof(Complex)*N_total*batch, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    gpuErrchk(cudaEventSynchronize(stop));

    cufftDestroy(fft_plan);
    gpuErrchk(cudaFree(in_one)); 
    gpuErrchk(cudaFree(in_two));

    // miliseconds
    float cufft_duration;
    cudaEventElapsedTime(&cufft_duration, start, stop); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return cufft_duration*1e-3;
}

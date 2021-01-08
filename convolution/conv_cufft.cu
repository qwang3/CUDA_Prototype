#include <cufft.h>
#include <cuda_runtime.h>

#include "main.hpp"

__global__ void dot(cufftComplex in_one[], cufftComplex in_two[], cufftComplex out[], int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        out[idx].x = in_one[idx].x * in_two[idx].x;
        out[idx].y = in_one[idx].y * in_two[idx].y;
        idx += blockDim.x;
    }
}

double convolve_cufft(float input_array_1[], float input_array_2[], 
            float output_array[], int N_max) {
    cufftComplex *in_one, *in_two, *tf_one, *tf_two, *out;
    cufftHandle fft_plan;
    int N_total = pow(N_max, 3);
    int n_block = 1;
    int n_thread = 1024;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    /* allocate memory */
    cudaMalloc(&in_one, sizeof(cufftComplex)*N_total);
    cudaMalloc(&in_two, sizeof(cufftComplex)*N_total);
    cudaMalloc(&tf_one, sizeof(cufftComplex)*N_total);
    cudaMalloc(&tf_two, sizeof(cufftComplex)*N_total);
    cudaMalloc(&out, sizeof(cufftComplex)*N_total);

    /* copy over input */
    printf("[CUFFT] copying input \n");
    cudaMemcpy2D((float*)in_one, 2*sizeof(float), 
                    input_array_1, sizeof(float), sizeof(float),
                N_total, cudaMemcpyHostToDevice);
    cudaMemcpy2D((float*)in_two, 2*sizeof(float), 
                    input_array_2, sizeof(float), sizeof(float),
                N_total, cudaMemcpyHostToDevice);

    printf("[CUFFT] computing, timer on... \n");
    cudaEventRecord(start);
    cufftPlan3d(&fft_plan, N_max, N_max, N_max, CUFFT_C2C);
    /* F = conv(f) */
    cufftExecC2C(fft_plan, in_one, tf_one, CUFFT_FORWARD);
    /* G = conv(g) */
    cufftExecC2C(fft_plan, in_two, tf_two, CUFFT_FORWARD);
    /* F dot G */
    dot<<<n_block, n_thread>>>(tf_one, tf_two, out, N_total);
    /* conv^-1(F dot G) */
    cufftExecC2C(fft_plan, out, out, CUFFT_INVERSE);

    cudaEventRecord(stop);
    printf("[CUFFT] compute finished, timer off... \n");

    cudaMemcpy2D(output_array, sizeof(float), 
                    out, 2*sizeof(float), sizeof(float),
                N_total, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    cufftDestroy(fft_plan);
    cudaFree(in_one); 
    cudaFree(in_two);
    cudaFree(tf_one); 
    cudaFree(tf_two);
    cudaFree(out);

    // miliseconds
    float cufft_duration;
    cudaEventElapsedTime(&cufft_duration, start, stop); 
    return cufft_duration*1e-3;
}

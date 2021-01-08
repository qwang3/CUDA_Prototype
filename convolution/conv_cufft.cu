#include <cufft.h>

#include "main.hpp"
#include "gpuErrchk.hpp"

__global__ void dot(cufftDoubleComplex in_one[], cufftDoubleComplex in_two[], cufftDoubleComplex out[], int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        // some attempt at ILP ...
        float in_one_x = in_one[idx].x;
        float in_one_y = in_one[idx].y;
        float in_two_x = in_two[idx].x;
        float in_two_y = in_two[idx].y;

        out[idx].x = in_one_x * in_two_x;
        out[idx].y = in_one_y * in_two_y;

        // out[idx].x = in_one[idx].x * in_two[idx].x;
        // out[idx].y = in_one[idx].y * in_two[idx].y;
        idx += blockDim.x;
    }
}

double convolve_cufft(double input_array_1[], double input_array_2[], 
            double output_array[], int N_max, int batch) {
    cufftDoubleComplex *in_one, *in_two, *tf_one, *tf_two, *out;
    cufftHandle fft_plan;
    int dim[3] = {N_max, N_max, N_max};
    int N_total = pow(N_max, 3);
    int n_block = 1;
    int n_thread = 1024;

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    
    /* allocate memory */
    gpuErrchk(cudaMalloc(&in_one, sizeof(cufftDoubleComplex)*N_total*batch));
    gpuErrchk(cudaMalloc(&in_two, sizeof(cufftDoubleComplex)*N_total*batch));
    gpuErrchk(cudaMalloc(&tf_one, sizeof(cufftDoubleComplex)*N_total*batch));
    gpuErrchk(cudaMalloc(&tf_two, sizeof(cufftDoubleComplex)*N_total*batch));
    gpuErrchk(cudaMalloc(&out, sizeof(cufftDoubleComplex)*N_total*batch));

    /* copy over input */
    printf("[CUFFT] copying input \n");
    gpuErrchk(cudaMemcpy2D((float*)in_one, 2*sizeof(float), 
                    input_array_1, sizeof(float), sizeof(float),
                    N_total*batch, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy2D((float*)in_two, 2*sizeof(float), 
                    input_array_2, sizeof(float), sizeof(float),
                    N_total*batch, cudaMemcpyHostToDevice));
    cufftPlanMany(&fft_plan, 3, dim, 
                    NULL, 0, 0,  
                    NULL, 0, 0, CUFFT_C2C, batch);

    printf("[CUFFT] computing, timer on... \n");
    cudaEventRecord(start);

    /* F = conv(f) */
    cufftExecZ2Z(fft_plan, in_one, tf_one, CUFFT_FORWARD);
    /* G = conv(g) */
    cufftExecZ2Z(fft_plan, in_two, tf_two, CUFFT_FORWARD);
    /* F dot G */
    dot<<<n_block, n_thread>>>(tf_one, tf_two, out, N_total*batch);
    /* conv^-1(F dot G) */
    cufftExecZ2Z(fft_plan, out, out, CUFFT_INVERSE);

    cudaEventRecord(stop);
    printf("[CUFFT] compute finished, timer off... \n");

    gpuErrchk(cudaMemcpy2D(output_array, sizeof(float), 
                    out, 2*sizeof(float), sizeof(float),
                    N_total*batch, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventSynchronize(stop));

    cufftDestroy(fft_plan);
    gpuErrchk(cudaFree(in_one)); 
    gpuErrchk(cudaFree(in_two));
    gpuErrchk(cudaFree(tf_one)); 
    gpuErrchk(cudaFree(tf_two));
    gpuErrchk(cudaFree(out));

    // miliseconds
    float cufft_duration;
    cudaEventElapsedTime(&cufft_duration, start, stop); 
    return cufft_duration*1e-3;
}

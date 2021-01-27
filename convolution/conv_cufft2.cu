#include <cufftw.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "main.hpp"
#include "gpuErrchk.hpp"

static __global__ void point_mul(cufftComplex *in_one, const cufftComplex *in_two, 
                                int N, int scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        cufftComplex in_1 = in_one[idx];
        cufftComplex in_2 = in_two[idx];
        cufftComplex out;

        out.x = (in_1.x * in_2.x - in_1.y * in_2.y)/scale;
        out.y = (in_1.x * in_2.y + in_1.y * in_2.x)/scale;
        in_one[idx] = out;
        idx += blockDim.x*gridDim.x;
    }
}

float convolve_cufft_3(Complex input_array_1[], Complex input_array_2[], 
            Complex output_array[], int N_max, int batch) {
    cufftComplex *in_one, *in_two;
    int N_total = pow(N_max, 3);

    // Registers host memory as page-locked (required for asynch cudaMemcpyAsync)
    gpuErrchk(cudaHostRegister(input_array_1, N_total*batch*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(input_array_2, N_total*batch*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(output_array, N_total*batch*sizeof(float2), cudaHostRegisterPortable));

    // Calculate optimal block and grid sizes
    int minGridSize, gridSize, blockSize;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, point_mul, 0, N_total*batch));
    gridSize = (N_total*batch + blockSize - 1) / blockSize; 

    // streams 
    const int num_streams = batch*2; 
    cudaStream_t stream[num_streams];
    cufftHandle *fft_plan = new cufftHandle[num_streams];
    for (int i=0; i<num_streams; i++) {
        gpuErrchk(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
        cufftPlan3d(&fft_plan[i], N_max, N_max, N_max, CUFFT_C2C);
        cufftSetStream(fft_plan[i], stream[i]);
    }

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    
    /* allocate memory */
    gpuErrchk(cudaMalloc(&in_one, sizeof(cufftComplex)*N_total*batch));
    gpuErrchk(cudaMalloc(&in_two, sizeof(cufftComplex)*N_total*batch));

    cudaProfilerStart();
    cudaEventRecord(start);
    // #pragma unroll
    for (int i=0; i<batch; i++) {
        int loc = i * N_total;
        gpuErrchk(cudaMemcpyAsync(&in_one[loc], &input_array_1[loc], sizeof(float2)*N_total, 
                                cudaMemcpyHostToDevice, stream[2*i]));
        cufftExecC2C(fft_plan[2*i], &in_one[loc], &in_one[loc], CUFFT_FORWARD);
        gpuErrchk(cudaMemcpyAsync(&in_two[loc], &input_array_2[loc], sizeof(float2)*N_total, 
                                cudaMemcpyHostToDevice, stream[2*i+1]));
        cufftExecC2C(fft_plan[2*i+1], &in_two[loc], &in_two[loc], CUFFT_FORWARD);
        gpuErrchk(cudaStreamSynchronize(stream[2*i+1]));
        point_mul<<<gridSize, blockSize, 0, stream[2*i]>>>(&in_one[i], &in_two[i], N_total, N_total);
        cufftExecC2C(fft_plan[2*i], &in_one[loc], &in_one[loc], CUFFT_INVERSE);
        gpuErrchk(cudaMemcpyAsync(&output_array[loc], &in_one[loc], sizeof(float2)*N_total, 
                                cudaMemcpyDeviceToHost, stream[2*i]));
    }

    cudaEventRecord(stop);
    gpuErrchk(cudaEventSynchronize(stop));
    cudaProfilerStop();

    gpuErrchk(cudaFree(in_one)); 
    gpuErrchk(cudaFree(in_two));

    for (int i=0; i<num_streams; i++) {
        gpuErrchk(cudaStreamDestroy(stream[i]));
        cufftDestroy(fft_plan[i]);
    }
    delete[] fft_plan;

    gpuErrchk(cudaHostUnregister(input_array_1));
    gpuErrchk(cudaHostUnregister(input_array_2));
    gpuErrchk(cudaHostUnregister(output_array));


    // miliseconds
    float cufft_duration;
    cudaEventElapsedTime(&cufft_duration, start, stop); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return cufft_duration*1e-3;
}


float convolve_cufft_2(Complex input_array_1[], Complex input_array_2[], 
    Complex output_array[], int N_max, int batch) {
    cufftComplex *in_one, *in_two;
    int N_total = pow(N_max, 3);

    // Registers host memory as page-locked (required for asynch cudaMemcpyAsync)
    gpuErrchk(cudaHostRegister(input_array_1, N_total*batch*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(input_array_2, N_total*batch*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(output_array, N_total*batch*sizeof(float2), cudaHostRegisterPortable));

    // Calculate optimal block and grid sizes
    int minGridSize, gridSize, blockSize;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, point_mul, 0, N_total*batch));
    gridSize = (N_total*batch + blockSize - 1) / blockSize; 

    // streams 
    const int num_streams = batch; 
    cudaStream_t stream[num_streams];
    cufftHandle *fft_plan = new cufftHandle[num_streams];
    for (int i=0; i<num_streams; i++) {
        gpuErrchk(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
        cufftPlan3d(&fft_plan[i], N_max, N_max, N_max, CUFFT_C2C);
        cufftSetStream(fft_plan[i], stream[i]);
    }

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    /* allocate memory */
    gpuErrchk(cudaMalloc(&in_one, sizeof(cufftComplex)*N_total*batch));
    gpuErrchk(cudaMalloc(&in_two, sizeof(cufftComplex)*N_total*batch));

    cudaEventRecord(start);
    for (int i=0; i<num_streams; i++) {
        int loc = i * N_total;
        gpuErrchk(cudaMemcpyAsync(&in_one[loc], &input_array_1[loc], sizeof(float2)*N_total, 
                                cudaMemcpyHostToDevice, stream[i]));
        cufftExecC2C(fft_plan[i], &in_one[loc], &in_one[loc], CUFFT_FORWARD);
    }
    for (int i=0; i<num_streams; i++) {
        int loc = i * N_total;
        gpuErrchk(cudaMemcpyAsync(&in_two[loc], &input_array_2[loc], sizeof(float2)*N_total, 
                                cudaMemcpyHostToDevice, stream[i]));
        cufftExecC2C(fft_plan[i], &in_two[loc], &in_two[loc], CUFFT_FORWARD);
    }
    for (int i=0; i<num_streams; i++) {
        gpuErrchk(cudaStreamSynchronize(stream[i]));
    }
    point_mul<<<gridSize, blockSize, 0, stream[0]>>>(in_one, in_two, N_total*batch, N_total);
    for (int i=0; i<num_streams; i++) {
        int loc = i * N_total;
        cufftExecC2C(fft_plan[i], &in_one[loc], &in_one[loc], CUFFT_INVERSE);
        gpuErrchk(cudaMemcpyAsync(&output_array[loc], &in_one[loc], sizeof(float2)*N_total, 
                            cudaMemcpyDeviceToHost, stream[i]));
    }
    cudaEventRecord(stop);
    gpuErrchk(cudaEventSynchronize(stop));

    gpuErrchk(cudaFree(in_one)); 
    gpuErrchk(cudaFree(in_two));
    for (int i=0; i<num_streams; i++) {
        gpuErrchk(cudaStreamDestroy(stream[i]));
        cufftDestroy(fft_plan[i]);
    }
    delete[] fft_plan;

    gpuErrchk(cudaHostUnregister(input_array_1));
    gpuErrchk(cudaHostUnregister(input_array_2));
    gpuErrchk(cudaHostUnregister(output_array));


    // miliseconds
    float cufft_duration;
    cudaEventElapsedTime(&cufft_duration, start, stop); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return cufft_duration*1e-3;
}


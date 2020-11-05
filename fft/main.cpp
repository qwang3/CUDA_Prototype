#include <ctime>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

#include <fftw3.h>

#include <cuda_runtime.h>
#include <cufft.h>

using namespace std;
typedef float2 Complex;

void usage(int argc, char** argv) {
    std::cout << "usage: " << argv[0] << " Nx Ny Nz Batch"<< std::endl;
    std::cout << "   Nx    " << "matrix dimension in x direction "<< std::endl;
    std::cout << "   Ny    " << "matrix dimension in y direction "<< std::endl;
    std::cout << "   Nz    " << "matrix dimension in z direction "<< std::endl;
    std::cout << "   Batch " << "number of matrices generated "<< std::endl;
}

int main(int argc, char** argv) {
    /* check input arguments */
    if (argc != 5) {
        usage(argc, argv);
        exit(1);
    }

    unsigned int Nx = atoi(argv[1]);
    unsigned int Ny = atoi(argv[2]);
    unsigned int Nz = atoi(argv[3]);
    unsigned int Batch = atoi(argv[4]);
    unsigned int total_size = Nx*Ny*Nz*Batch;

    float *data = new float[total_size];
    for (int i=0; i<total_size; i++) {
        data[i] = rand() / (float)RAND_MAX; 
    }

    // /* copy data to fftw */
    // fftw_complex *fft_data = reinterpret_cast<fftw_complex>(data);


    // /* create FFTW plan */
    // fftw_plan_dft_3d(Nx, Ny, Nz, )

    /* allocate array on device */
    float *data_cuda;
    cudaMalloc((void**) &data_cuda, sizeof(float) * total_size);
    /* Copy data from Host Device  */
    cudaMemcpy(data_cuda, data, sizeof(float) * total_size, cudaMemcpyHostToDevice);

    /* cufftComplex  data type*/
    cufftComplex *cuda_idata, *cuda_odata, *cuda_fdata;
    
    cudaMalloc((void**) &cuda_idata, sizeof(cufftComplex) * total_size);
    cudaMalloc((void**) &cuda_odata, sizeof(cufftComplex) * total_size));
    cudaMalloc((void**) &cuda_fdata, sizeof(cufftComplex) * total_size));

    // 4 blocks each with 32 threads 
    real2complex<4, 32>

    /* Randomly generate data */
    for (unsigned int i = 0; i < total_size; i++) {
        data[i].x = 1;    // rand() / (float)RAND_MAX;
        data[i].y = 0;
    }
    cudaMemcpy(cuda_idata, data, memory_size, cudaMemcpyHostToDevice);

    /* create CUDA fft plan */
    /* NOTE: can potentially use cufftPlanMany for more advanced batch layout */
    cufftHandle cuda_plan;
    cufftPlan3d(&cuda_plan, Nx, Ny, Nz, CUFFT_C2C);

    /* transform */
    for (int i = 0; i < Batch; i++) {
        cufftExecC2C(cuda_plan, cuda_idata, cuda_idata, CUFFT_FORWARD);
    }
    /* trasform back */
    for (int i = 0; i < Batch; i++) {
        cufftExecC2C(cuda_plan, cuda_odata, cuda_fdata, CUFFT_INVERSE);
    }
    /* validation */
    Complex *fdata = new Complex[total_size];
    cudaMemcpy(fdata, cuda_fdata, memory_size, cudaMemcpyDeviceToHost);
    for (int i=0; i < total_size; i++) {
        if (data[i].x != fdata[i].x) {
            cout << "[ERROR] data[" <<i<<"] diverge. Data: " << data[i].x << " fdata: "<< fdata[i].x << endl;
        }
    }
}

/* convert a standard array of floats into a cufftComplex data type */ 
__global__ void real2complex(float*f,cufftComplex*fc,int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j=threadIdx.y + blockIdx.y * blockDim.y;
    int index=j * N + i;
    if(i < N && j < N) {
        fc[index].x = f[index];
        fc[index].y=0.0f;
    }
}

/* convert a cufftComplex into a standard float array */ 
__global__ void complex2real(cufftComplex *fc, float *f, int N) {
    int i = threadIdx.x + blockIdx.x * BSZ;
    int j = threadIdx.y + blockIdx.y * BSZ;
    int index = j * N + i;
    if(i < N && j < N) {
        f[index] = fc[index].x/((float)N*(float)N);
    }
}


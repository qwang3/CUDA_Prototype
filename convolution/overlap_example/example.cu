#include <stdio.h>

#include <cufft.h>

#define BLOCKSIZE 32
#define NUM_STREAMS 3

/**********/
/* iDivUp */
/*********/
int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/******************/
/* SUMMING KERNEL */
/******************/
__global__ void kernel(float2 *vec1, float2 *vec2, float2 *vec3, float2 *out, int N) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        out[tid].x = vec1[tid].x + vec2[tid].x + vec3[tid].x;
        out[tid].y = vec1[tid].y + vec2[tid].y + vec3[tid].y;
    }

}


/********/
/* MAIN */
/********/
int main()
{
    const int N = 600000;
    const int Npartial = N / NUM_STREAMS;

    // --- Host input data initialization
    float2 *h_in1 = new float2[Npartial];
    float2 *h_in2 = new float2[Npartial];
    float2 *h_in3 = new float2[Npartial];
    for (int i = 0; i < Npartial; i++) {
        h_in1[i].x = 1.f;
        h_in1[i].y = 0.f;
        h_in2[i].x = 1.f;
        h_in2[i].y = 0.f;
        h_in3[i].x = 1.f;
        h_in3[i].y = 0.f;
    }

    // --- Host output data initialization
    float2 *h_out = new float2[N];

    // --- Registers host memory as page-locked (required for asynch cudaMemcpyAsync)
    gpuErrchk(cudaHostRegister(h_in1, Npartial*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(h_in2, Npartial*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(h_in3, Npartial*sizeof(float2), cudaHostRegisterPortable));

    // --- Device input data allocation
    float2 *d_in1;          gpuErrchk(cudaMalloc((void**)&d_in1, N*sizeof(float2)));
    float2 *d_in2;          gpuErrchk(cudaMalloc((void**)&d_in2, N*sizeof(float2)));
    float2 *d_in3;          gpuErrchk(cudaMalloc((void**)&d_in3, N*sizeof(float2)));
    float2 *d_out1;         gpuErrchk(cudaMalloc((void**)&d_out1, N*sizeof(float2)));
    float2 *d_out2;         gpuErrchk(cudaMalloc((void**)&d_out2, N*sizeof(float2)));
    float2 *d_out3;         gpuErrchk(cudaMalloc((void**)&d_out3, N*sizeof(float2)));
    float2 *d_out;          gpuErrchk(cudaMalloc((void**)&d_out, N*sizeof(float2)));

    // --- Zero padding
    gpuErrchk(cudaMemset(d_in1, 0, N*sizeof(float2)));
    gpuErrchk(cudaMemset(d_in2, 0, N*sizeof(float2)));
    gpuErrchk(cudaMemset(d_in3, 0, N*sizeof(float2)));

    // --- Creates CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) gpuErrchk(cudaStreamCreate(&streams[i]));

    // --- Creates cuFFT plans and sets them in streams
    cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cufftPlan1d(&plans[i], N, CUFFT_C2C, 1);
        cufftSetStream(plans[i], streams[i]);
    }

    // --- Async memcopyes and computations
    gpuErrchk(cudaMemcpyAsync(d_in1, h_in1, Npartial*sizeof(float2), cudaMemcpyHostToDevice, streams[0]));
    gpuErrchk(cudaMemcpyAsync(&d_in2[Npartial], h_in2, Npartial*sizeof(float2), cudaMemcpyHostToDevice, streams[1]));
    gpuErrchk(cudaMemcpyAsync(&d_in3[2*Npartial], h_in3, Npartial*sizeof(float2), cudaMemcpyHostToDevice, streams[2]));
    cufftExecC2C(plans[0], (cufftComplex*)d_in1, (cufftComplex*)d_out1, CUFFT_FORWARD);
    cufftExecC2C(plans[1], (cufftComplex*)d_in2, (cufftComplex*)d_out2, CUFFT_FORWARD);
    cufftExecC2C(plans[2], (cufftComplex*)d_in3, (cufftComplex*)d_out3, CUFFT_FORWARD);

    for(int i = 0; i < NUM_STREAMS; i++) gpuErrchk(cudaStreamSynchronize(streams[i]));

    kernel<<<iDivUp(BLOCKSIZE,N), BLOCKSIZE>>>(d_out1, d_out2, d_out3, d_out, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_out, d_out, N*sizeof(float2), cudaMemcpyDeviceToHost));

    // for (int i=0; i<N; i++) printf("i = %i; real(h_out) = %f; imag(h_out) = %f\n", i, h_out[i].x, h_out[i].y);

    // --- Releases resources
    gpuErrchk(cudaHostUnregister(h_in1));
    gpuErrchk(cudaHostUnregister(h_in2));
    gpuErrchk(cudaHostUnregister(h_in3));
    gpuErrchk(cudaFree(d_in1));
    gpuErrchk(cudaFree(d_in2));
    gpuErrchk(cudaFree(d_in3));
    gpuErrchk(cudaFree(d_out1));
    gpuErrchk(cudaFree(d_out2));
    gpuErrchk(cudaFree(d_out3));
    gpuErrchk(cudaFree(d_out));

    for(int i = 0; i < NUM_STREAMS; i++) gpuErrchk(cudaStreamDestroy(streams[i]));

    delete[] h_in1;
    delete[] h_in2;
    delete[] h_in3;
    delete[] h_out;

    cudaDeviceReset();  

    return 0;
}
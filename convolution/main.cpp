#include <cstring>
#include <cstdlib>
#include <cassert>

#include "main.hpp"

/* print out a usage message */
void usage(int argc, char **argv) {
    fprintf(stderr, "usage: %s n_max batch n_omp_threads\n", argv[0]);
}

/* initialize a 3D input array, row major formatted */
void init_input(float *input_array_1, float *input_array_2, int N_max, int batch) {
    
    int N_total = pow(N_max, 3) * batch;
    #pragma omp parallel for
    for (int i=0; i< N_total; i++) {
        input_array_1[i] = (float) rand()/RAND_MAX;
        input_array_2[i] = (float) rand()/RAND_MAX;
    }
}

int main(int argc, char **argv) {

    if (argc != 4) {
        usage(argc, argv);
        return 1;
    }

    int N_max = atoi(argv[1]);
    int N_batch = atoi(argv[2]);
    int n_threads = atoi(argv[3]);
    int N_total = pow(N_max, 3)*N_batch;

    float *input_array_1 = new float[N_total];
    float *input_array_2 = new float[N_total];
    float *output_fftw = new float[N_total];
    float *output_cufft = new float[N_total];
    init_input(input_array_1, input_array_2, N_max, N_batch);
    omp_set_num_threads(n_threads);

    double fftw_time = convolve_fftw(input_array_1, input_array_2, output_fftw, N_max, N_batch);
    float cufft_time = convolve_cufft(input_array_1, input_array_2, output_cufft, N_max, N_batch);

    //verify results
    // #pragma omp parallel for
    for (int i = 0; i<N_total; i++) {
        if (abs(output_cufft[i] - output_fftw[i]) > 1e3) {
            // continue;
            printf("Error at: cufft[%d]: %f ",i, output_cufft[i]);
            printf("fftw[%d]: %f \n",i, output_fftw[i]);
        }
    }

    printf("[FFTW] took %e s per input cell \n", fftw_time/N_total/N_batch);
    printf("[CUFFT] took %e s per input cell \n", cufft_time/N_total/N_batch);

    return 0;
}
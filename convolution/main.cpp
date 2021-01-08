#include <cstring>
#include <cstdlib>
#include <cassert>

#include "main.hpp"

/* print out a usage message */
void usage(int argc, char **argv) {
    fprintf(stderr, "usage: %s n_max n_omp_threads\n", argv[0]);
}

/* initialize a 3D input array, row major formatted */
void init_input(float *input_array_1, float *input_array_2, int N_max) {
    
    int N_total = pow(N_max, 3);
    #pragma omp parallel for
    for (int i=0; i< N_total; i++) {
        input_array_1[i] = 1;
        input_array_2[i] = 1;
    }
}

int main(int argc, char **argv) {

    if (argc != 3) {
        usage(argc, argv);
        return 1;
    }

    int N_max = atoi(argv[1]);
    int n_threads = atoi(argv[2]);
    int N_total = pow(N_max, 3);

    float *input_array_1 = new float[N_total];
    float *input_array_2 = new float[N_total];
    float *output_fftw = new float[N_total];
    float *output_cufft = new float[N_total];
    init_input(input_array_1, input_array_2, N_max);
    omp_set_num_threads(n_threads);

    double fftw_time = convolve_fftw(input_array_1, input_array_2, output_fftw, N_max);
    float cufft_time = convolve_cufft(input_array_1, input_array_2, output_cufft, N_max);

    //verify results
    #pragma omp parallel for
    for (int i = 0; i<N_total; i++) {
        // printf("cufft[%d]: %f ",i, output_cufft[i]);
        // printf("fftw[%d]: %f \n",i, output_fftw[i]);
        assert(output_cufft[i] == output_fftw[i]);
    }

    printf("[FFTW] took %e s per input cell \n", fftw_time/N_total);
    printf("[CUFFT] took %e s per input cell \n", cufft_time/N_total);

    return 0;
}
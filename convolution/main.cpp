#include <cstring>
#include <cstdlib>
#include <cassert>

#include "main.hpp"

/* print out a usage message */
void usage(int argc, char **argv) {
    fprintf(stderr, "usage: %s n_max batch n_omp_threads\n", argv[0]);
}

/* initialize a 3D input array, row major formatted */
void init_input(Complex *input_array_1, Complex *input_array_2, int N_max, int batch) {
    
    int N_total = pow(N_max, 3) * batch;
    #pragma omp parallel for
    for (int i=0; i< N_total; i++) {
        input_array_1[i].x = (float) rand()/RAND_MAX;
        input_array_2[i].x = (float) rand()/RAND_MAX;
        input_array_1[i].y = (float) rand()/RAND_MAX;
        input_array_2[i].y = (float) rand()/RAND_MAX;
        // input_array_1[i].x = i;
        // input_array_2[i].x = i;
        // input_array_1[i].y = 0;
        // input_array_2[i].y = 0;
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

    Complex *input_array_1 = new Complex[N_total];
    Complex *input_array_2 = new Complex[N_total];
    Complex *output_fftw = new Complex[N_total];
    Complex *output_cufft = new Complex[N_total];
    init_input(input_array_1, input_array_2, N_max, N_batch);
    omp_set_num_threads(n_threads);

    float fftw_time = convolve_fftw(input_array_1, input_array_2, output_fftw, N_max, N_batch);
    float cufft_time_2 = convolve_cufft_1(input_array_1, input_array_2, output_cufft, N_max, N_batch);
    // float cufft_time_3 = convolve_cufft_3(input_array_1, input_array_2, output_cufft, N_max, N_batch);

    //verify results
    // #pragma omp parallel for
    for (int i = 0; i<N_total; i++) {
        float err_x = fabs(output_cufft[i].x - output_fftw[i].x);
        float err_y = fabs(output_cufft[i].y - output_fftw[i].y);
        if (err_x > 1e-3) {
            // continue;
            printf("Error at: cufft[%d].x: %f ",i, output_cufft[i].x);
            printf("fftw[%d].x: %f ",i, output_fftw[i].x);
            printf("Err_x: %f \n", err_x);
        }
        if (err_y > 1e-3) {
            // continue;
            printf("Error at: cufft[%d].y: %f ",i, output_cufft[i].y);
            printf("fftw[%d].y: %f ",i, output_fftw[i].y);
            printf("Err_y: %f \n", err_y);
        }
    }

    printf("[FFTW] took %e s per input cell \n", fftw_time/N_total/N_batch);
    printf("[CUFFT2] took %e s per input cell \n", cufft_time_2/N_total/N_batch);
    // printf("[CUFFT3] took %e s per input cell \n", cufft_time_3/N_total/N_batch);

    return 0;
}
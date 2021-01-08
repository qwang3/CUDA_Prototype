#include <fftw3.h>

#include "main.hpp"

double convolve_fftw(float input_array_1[], float input_array_2[], 
            float output_array[], int N_max, int batch) {

    fftw_complex *in_one, *in_two, *tf_one, *tf_two, *out;
    fftw_plan forward_fft1, forward_fft2, inverse_fft;
    int N_total = pow(N_max, 3);
    int dim[3] = {N_max, N_max, N_max};

    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    printf("[FFTW] Starting with %d thread(s)\n", omp_get_max_threads());

    
    /* allocate memory */
    printf("[FFTW] Allocating memory \n");
    in_one = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_total*batch);
    in_two = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_total*batch);
    tf_one = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_total*batch);
    tf_two = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_total*batch);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_total*batch);

    /* copy over input */
    printf("[FFTW] copying input \n");
    #pragma omp parallel for
    for (int i = 0; i < N_total*batch; i++) {
        // RE
        in_one[i][0] = input_array_1[i];
        in_two[i][0] = input_array_2[i];
        // IM
        in_one[i][1] = 0;
        in_two[i][1] = 0;
    }

    forward_fft1 = fftw_plan_many_dft(3, dim, batch, 
                                    in_one, dim, 1, N_total, 
                                    tf_one, dim, 1, N_total, 
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    forward_fft2 = fftw_plan_many_dft(3, dim, batch, 
                                    in_two, dim, 1, N_total, 
                                    tf_two, dim, 1, N_total, 
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    inverse_fft = fftw_plan_many_dft(3, dim, batch, 
                                    out, dim, 1, N_total, 
                                    out, dim, 1, N_total, 
                                    FFTW_BACKWARD, FFTW_ESTIMATE);

    printf("[FFTW] computing, timer on... \n");
    double tic = omp_get_wtime();
    /* F = conv(f) */
    fftw_execute(forward_fft1);
    /* G = conv(g) */
    fftw_execute(forward_fft2);
    /* F dot G */
    #pragma omp parallel for
    for (int i=0; i<N_total*batch; i++) {
        out[i][0] = tf_one[i][0] * tf_two[i][0];
        out[i][1] = tf_one[i][1] * tf_two[i][1];
    }
    /* conv^-1(F dot G) */
    fftw_execute(inverse_fft);

    double toc = omp_get_wtime();
    printf("[FFTW] compute finished, timer off... \n");

    /* copy over output */
    for (int i = 0; i < N_total*batch; i++) {
        // RE
        output_array[i] = out[i][0];
    }

    fftw_destroy_plan(forward_fft1);
    fftw_destroy_plan(forward_fft2);
    fftw_destroy_plan(inverse_fft);
    fftw_free(in_one); 
    fftw_free(in_two);
    fftw_free(tf_one); 
    fftw_free(tf_two);
    fftw_free(out);
    fftw_cleanup_threads();

    return toc - tic;
}
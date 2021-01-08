#include <fftw3.h>

#include "main.hpp"

double convolve_fftw(float input_array_1[], float input_array_2[], 
            float output_array[], int N_max) {

    fftw_complex *in_one, *in_two, *tf_one, *tf_two, *out;
    fftw_plan forward_fft1, forward_fft2, inverse_fft;
    int N_total = pow(N_max, 3);

    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    
    /* allocate memory */
    printf("[FFTW] Allocating memory \n");
    in_one = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_total);
    in_two = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_total);
    tf_one = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_total);
    tf_two = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_total);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_total);

    /* copy over input */
    printf("[FFTW] copying input \n");
    #pragma omp parallel for
    for (int i = 0; i < N_total; i++) {
        // RE
        in_one[i][0] = input_array_1[i];
        in_two[i][0] = input_array_2[i];
        // IM
        in_one[i][1] = 0;
        in_two[i][1] = 0;
    }

    printf("[FFTW] computing, timer on... \n");
    double tic = omp_get_wtime();
    /* F = conv(f) */
    forward_fft1 = fftw_plan_dft_3d(N_max, N_max, N_max, in_one, tf_one, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(forward_fft1);
    /* G = conv(g) */
    forward_fft2 = fftw_plan_dft_3d(N_max, N_max, N_max, in_two, tf_two, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(forward_fft2);
    /* F dot G */
    #pragma omp parallel for
    for (int i=0; i<N_total; i++) {
        out[i][0] = tf_one[i][0] * tf_two[i][0];
        out[i][1] = tf_one[i][1] * tf_two[i][1];
    }
    /* conv^-1(F dot G) */
    inverse_fft = fftw_plan_dft_3d(N_max, N_max, N_max, out, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(inverse_fft);
    double toc = omp_get_wtime();
    printf("[FFTW] compute finished, timer off... \n");

    /* copy over output */
    for (int i = 0; i < N_total; i++) {
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

    // seconds
    return (toc - tic); 
}
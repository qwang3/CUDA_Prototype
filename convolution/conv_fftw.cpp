#include <fftw3.h>

#include "main.hpp"

void print_array(fftwf_complex *array, const int size) {
    for (int i=0; i<size; i++) {
        if (array[i][0] != 0) {
            printf("    index %d: arr.x= %f, arr.y=%f \n", i, array[i][0], array[i][1]);
        }
        
    }
}

float convolve_fftw(Complex input_array_1[], Complex input_array_2[], 
            Complex output_array[], int N_max, int batch) {

    fftwf_complex *in_one, *in_two, *tf_one, *tf_two, *out;
    fftwf_plan forward_fft1, forward_fft2, inverse_fft;
    int N_total = pow(N_max, 3);
    int dim[3] = {N_max, N_max, N_max};

    
    /* allocate memory */
    in_one = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N_total*batch);
    in_two = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N_total*batch);
    tf_one = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N_total*batch);
    tf_two = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N_total*batch);
    out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N_total*batch);

    double tic = omp_get_wtime();
    for (int i = 0; i < N_total*batch; i++) {
        // RE
        in_one[i][0] = input_array_1[i].x;
        in_two[i][0] = input_array_2[i].x;
        // IM
        in_one[i][1] = input_array_1[i].y;
        in_two[i][1] = input_array_2[i].y;
    }

    forward_fft1 = fftwf_plan_many_dft(3, dim, batch, 
                                    in_one, dim, 1, N_total, 
                                    tf_one, dim, 1, N_total, 
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    forward_fft2 = fftwf_plan_many_dft(3, dim, batch, 
                                    in_two, dim, 1, N_total, 
                                    tf_two, dim, 1, N_total, 
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    inverse_fft = fftwf_plan_many_dft(3, dim, batch, 
                                    out, dim, 1, N_total, 
                                    out, dim, 1, N_total, 
                                    FFTW_BACKWARD, FFTW_ESTIMATE);
    /* F = conv(f) */
    fftwf_execute(forward_fft1);
    /* G = conv(g) */
    fftwf_execute(forward_fft2);
    /* F dot G  and scale*/
    for (int i=0; i<N_total*batch; i++) {
        out[i][0] = (tf_one[i][0] * tf_two[i][0] - tf_one[i][1] * tf_two[i][1])/N_total;
        out[i][1] = (tf_one[i][0] * tf_two[i][1] + tf_one[i][1] * tf_two[i][0])/N_total;
        // printf("fftw index %d: out.x = %f out.y = %f \n", i,out[i][0], out[i][1]);
    }
    /* conv^-1(F dot G) */
    // printf("### [fftw] Printing final result before fft \n");
    // print_array(out, N_total*batch);
    fftwf_execute(inverse_fft);
    // printf("### [ffw] Printing final result after fft\n");
    // print_array(out, N_total*batch);

    /* copy over output */
    for (int i = 0; i < N_total*batch; i++) {
        output_array[i].x = (float) out[i][0]; //RE
        output_array[i].y = (float) out[i][1]; //IM
    }
    double toc = omp_get_wtime();

    fftwf_destroy_plan(forward_fft1);
    fftwf_destroy_plan(forward_fft2);
    fftwf_destroy_plan(inverse_fft);
    fftwf_free(in_one); 
    fftwf_free(in_two);
    fftwf_free(tf_one); 
    fftwf_free(tf_two);
    fftwf_free(out);

    return toc - tic;
}
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <string>
#include <fstream>

#include "main.hpp"

/* print out a usage message */
void usage(int argc, char **argv) {
    fprintf(stderr, "usage: %s filename dim, max_batch\n", argv[0]);
}

/* initialize a 3D input array, row major formatted */
void init_input(Complex *input_array_1, Complex *input_array_2, int N_max, int batch) {
    
    int N_total = pow(N_max, 3) * batch;
    #pragma omp parallel for
    for (int i=0; i< N_total; i++) {
        // input_array_1[i].x = (float) rand()/RAND_MAX;
        // input_array_2[i].x = (float) rand()/RAND_MAX;
        // input_array_1[i].y = (float) rand()/RAND_MAX;
        // input_array_2[i].y = (float) rand()/RAND_MAX;
        input_array_1[i].x = 1;
        input_array_2[i].x = 1;
        input_array_1[i].y = 0;
        input_array_2[i].y = 0;
    }
}

int main(int argc, char **argv) {

    if (argc != 4) {
        usage(argc, argv);
        return 1;
    }

    std::string filename = argv[1];
    int N_max = atoi(argv[2]);
    int N_batch = atoi(argv[3]);

    std::ofstream result_file;
    char line[100];
    memset(line, 0, sizeof(char) * 100);
    result_file.open(filename);
    result_file << "Batch Size, FFTW (s), cufft_1 (s), cufft_2 (s), cufft_3 (s) \n";
    
    for (int batch=1; batch < N_batch; batch++) {
        printf("Running %d inputs \n", batch);
        int N_total = pow(N_max, 3)*batch;

        //input 
        Complex *input_array_1 = new Complex[N_total];
        Complex *input_array_2 = new Complex[N_total];
        Complex *output_fftw = new Complex[N_total];
        Complex *output_cufft_1 = new Complex[N_total];
        Complex *output_cufft_2 = new Complex[N_total];
        Complex *output_cufft_3 = new Complex[N_total];
        init_input(input_array_1, input_array_2, N_max, batch);
        omp_set_num_threads(1);

        float fftw_time = convolve_fftw(input_array_1, input_array_2, output_fftw, N_max, batch);
        float cufft_time_1 = convolve_cufft_1(input_array_1, input_array_2, output_cufft_1, N_max, batch);
        float cufft_time_2 = convolve_cufft_2(input_array_1, input_array_2, output_cufft_2, N_max, batch);
        float cufft_time_3 = convolve_cufft_3(input_array_1, input_array_2, output_cufft_3, N_max, batch);
    
        //verify resultscompare
        // #pragma omp parallel for
        for (int i = 0; i<N_total; i++) {
            float err_x = abs(output_cufft_1[i].x - output_fftw[i].x);
            float err_y = abs(output_cufft_1[i].y - output_fftw[i].y);
            if (err_x > 1e-3) {
                // continue;
                printf("Error at: cufft[%d].x: %f ",i, output_cufft_1[i].x);
                printf("fftw[%d].x: %f ",i, output_fftw[i].x);
                printf("Err: %f \n", err_x);
            }
            if (err_y > 1e-3) {
                // continue;
                printf("Error at: cufft[%d].y: %f ",i, output_cufft_1[i].y);
                printf("fftw[%d].y: %f ",i, output_fftw[i].y);
                printf("Err: %f \n", err_y);
            }
        }
        sprintf(line, "%d,%f,%f,%f,%f\n", batch,fftw_time, cufft_time_1, cufft_time_2, cufft_time_3);
        result_file << line;
        memset(line, 0, sizeof(char) * 100);
        
        delete[] input_array_1;
        delete[] input_array_2;
        delete[] output_cufft_1;
        delete[] output_cufft_2;
        delete[] output_cufft_3;
        delete[] output_fftw;
    }
    result_file.close();
    return 0;
}
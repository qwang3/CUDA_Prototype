#include "1dfft.hpp"

using namespace std;


/* Given a 1d array size, compute forward fft with both CPU and GPU, and return the time they took */
void benchmark(int Nx, float* time) {
    /* randomly generate data */


    double *idata = new double[Nx];
    double *odata_cpu = new double[Nx];
    double *odata_gpu = new double[Nx];
    for (int i=0; i<Nx; i++) {
        idata[i] = rand()/(double)RAND_MAX;
    }

    /* start the time for CPU now */
    time[0] = fft_fftw(idata, odata_cpu, Nx);

    /* start the time for GPU */
    time[1] = fft_cuda(idata, odata_gpu, Nx);

    /* NOTE: For some reason cuda suffer from precision issues. The higher the number, the 
     * less precise it is*/
    for (int i=0; i<Nx; i++) {
        if (abs(odata_cpu[i] - odata_gpu[i]) > 1e-3) {
            fprintf(stderr, "[ERROR] data at %d dissagree. fftw = %f, cuda = %f \n", 
                i, odata_cpu[i], odata_gpu[i]);
        }
    }
    delete[] idata;
    delete[] odata_cpu;
    delete[] odata_gpu;
}

void usage(int argc, char **argv) {
    cout << "usage: " << argv[0] << " Nmax" << endl;
    cout << "Nmax (int): maximum number of element in array. must be a exponent of 10" << endl;
}

int main(int argc, char **argv) {

    if (argc != 2) {
        usage(argc, argv);
        exit(1);
    }
    int Nmax = atoi(argv[1]);
    int Nruns = log10(Nmax);
    printf("Nruns: %d \n", Nruns);
    float **times = new float*[Nruns]; // cpu first column, gpu second column
    for (int i = 0; i < Nruns; i++) {
        times[i] = new float[2];
    }

    /* Actual benchmarking*/
    int index = 0;
    for (int Nx = 1; Nx < Nmax; Nx*= 10) {
        printf("starting: %d \n", index);
        benchmark(Nx, times[index]);
        index++;
    }

    /* write result to file */
    ofstream res_file; 
    res_file.open("result.csv");
    res_file << "Size, CPU_time, GPU_time,\n";
    for (int i = 0; i < Nruns; i++) {
        res_file << pow(10, i) << "," << times[i][0] << "," << times[i][1] << ",\n";
    }
    res_file.close();
    
    /* clean up */
    for (int i=0; i<Nruns; i++) {
        delete [] times[i];
    }
    delete [] times;

    return 0;

    /* result 
     * CPU outperforms GPU at Nx <= 1e7
     * GPU outperforms CPU at Nx >= 1e8
     */
}

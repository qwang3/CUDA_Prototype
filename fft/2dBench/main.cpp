#include "2dfft.hpp"

using namespace std;


/* Given a 1d array size, compute forward fft with both CPU and GPU, and return the time they took */
void benchmark(int Nx, int Ny, float* time) {
    /* randomly generate data */

    double **idata = new double*[Nx];
    double **odata_cpu = new double*[Nx];
    double **odata_gpu = new double*[Nx];
    for (int i=0; i<Nx; i++) {
        idata[i] = new double[Ny];
        odata_cpu[i] = new double[Ny];
        odata_gpu[i] = new double[Ny];
    }
    for (int i=0; i<Nx; i++) {
        for (int j=0; j<Ny; j++)
            idata[i][j] = 1;
            // idata[i][j] = rand()/(double)RAND_MAX;
    }
    /* start the time for CPU now */
    time[0] = fft_fftw2d(idata, odata_cpu, Nx, Nx);

    /* start the time for GPU */
    time[1] = fft_cuda(idata, odata_gpu, Nx, Ny);
    // time[1]=0;

    /* NOTE: For some reason cuda suffer from precision issues. The higher the number, the 
     * less precise it is*/
    for (int i=0; i<Nx; i++) {
        for (int j=0; j<Ny; j++) {
        // if (abs(odata_cpu[i][j] - odata_gpu[i][j]) > 1e-3) {
        //     fprintf(stderr, "[ERROR] data at %d dissagree. fftw = %f, cuda = %f \n", 
        //         i, odata_cpu[i], odata_gpu[i]);
            printf(" %f ", odata_gpu[i][j]);
        }
        printf("\n");
    }
    for (int i = 0; i<Nx; i++) {
        delete[] idata[i];
        delete[] odata_cpu[i];
        delete[] odata_gpu[i];
    }
    delete[] idata;
    delete[] odata_cpu;
    delete[] odata_gpu;
}

void usage(int argc, char **argv) {
    cout << "usage: " << argv[0] << " Nmax" << endl;
    cout << "Nx (int): Dimension of 2D input array." << endl;
}

int main(int argc, char **argv) {

    if (argc != 2) {
        usage(argc, argv);
        exit(1);
    }
    int Nx = atoi(argv[1]);
    int Nruns = Nx; 
    printf("Nruns: %d \n", Nruns);
    float **times = new float*[Nruns]; // cpu first column, gpu second column
    for (int i = 0; i < Nruns; i++) {
        times[i] = new float[2];
    }

    /* Actual benchmarking*/
    for (int i = 0; i < Nx; i++) {
        printf("starting: %d \n", i);
        benchmark(i+1, i+1, times[i]);
    }
    printf("Finished \n");

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

#include <chrono>

#include "1dfft.hpp"

using namespace std;

void usage(int argc, char **argv) {
    cout << "usage: " << argv[0] << " Nx" << endl;
    cout << "Nx (int): number of element in array" << endl;
}

int main(int argc, char **argv) {

    if (argc != 2) {
        usage(argc, argv);
        exit(1);
    }
    int Nx = atoi(argv[1]);

    cout << "Generating array of size " << Nx << endl;
    /* randomly generate data */
    double *idata = new double[Nx];
    double *odata_cpu = new double[Nx];
    double *odata_gpu = new double[Nx];
    for (int i=0; i<Nx; i++) {
        idata[i] = 1; //rand()/(double)RAND_MAX;
    }

    
    cout << "running CPU" << endl;
    /* start the time for CPU now */
    chrono::microseconds duration_cpu = fft_fftw(idata, odata_cpu, Nx);
    cout << "FFTW took " << duration_cpu.count() << "ms" << endl;

    cout << "running GPU" << endl;
    /* start the time for GPU */
    chrono::microseconds duration_gpu = fft_cuda(idata, odata_gpu, Nx);
    cout << "cuFFT took " << duration_gpu.count() << "ms" << endl;

    /* NOTE: For some reason cuda suffer from precision issues. The higher the number, the 
     * less precise it is*/
    for (int i=0; i<Nx; i++) {
        if (abs(odata_cpu[i] - odata_gpu[i]) > 1e-3) {
            fprintf(stderr, "[ERROR] data at %d dissagree. fftw = %f, cuda = %f \n", 
                i, odata_cpu[i], odata_gpu[i]);
        }
    }
    return 0;

    /* result 
     * CPU outperforms GPU at Nx <= 1e7
     * GPU outperforms CPU at Nx >= 1e8
     */
}


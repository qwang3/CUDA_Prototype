#include "2dfft.hpp"

using namespace std;

float fft_fftw2d(double** idata, double** odata, int Nx, int Ny) {

    /* convert idata to fftw_complex */
    fftw_complex *idata_c, *odata_c;
    idata_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    odata_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx * Ny);

    
    /* copy idata content */
    /* row major order */
    for (int i=0; i<Nx; i++) {
        for (int j=0; j<Ny; j++) {
            idata_c[i*Nx + j][0] = idata[i][j];
            idata_c[i*Nx + j][1] =0;
        }
    }

    /* run plan */
    fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, idata_c, odata_c, FFTW_FORWARD, FFTW_ESTIMATE);

    auto start = chrono::high_resolution_clock::now();
    fftw_execute(plan);
    auto finish = chrono::high_resolution_clock::now();
    chrono::nanoseconds duration_nano = chrono::duration_cast<chrono::nanoseconds>(finish - start);
    float duration = duration_nano.count() * 1e-6;


    /* copy idata content */
    for (int i=0; i<Nx; i++) {
        for (int j=0; j<Ny; j++) {
            odata[i][j] = odata_c[i*Nx+j][0];
        }

    }

    fftw_destroy_plan(plan);
    fftw_free(idata_c); 
    fftw_free(odata_c);

    return duration;
}
#include "1dfft.hpp"

using namespace std;

chrono::microseconds fft_fftw(const double* idata, double* odata, int Nx) {

    /* convert idata to fftw_complex */
    fftw_complex *idata_c, *odata_c;
    idata_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx);
    odata_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx);

    
    /* copy idata content */
    for (int i=0; i<Nx; i++) {
        idata_c[i][0] = idata[i];
        idata_c[i][1] = 0;
    }

    /* run plan */
    fftw_plan plan = fftw_plan_dft_1d(Nx, idata_c, odata_c, FFTW_FORWARD, FFTW_ESTIMATE);

    auto start = chrono::high_resolution_clock::now();
    fftw_execute(plan);
    auto finish = chrono::high_resolution_clock::now();
    chrono::microseconds duration = chrono::duration_cast<chrono::milliseconds>(start - finish);


    /* copy idata content */
    for (int i=0; i<Nx; i++) {
        odata[i]= odata_c[i][0];
    }

    fftw_destroy_plan(plan);
    fftw_free(idata_c); 
    fftw_free(odata_c);

    return duration;
}
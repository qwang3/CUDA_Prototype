#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

void usage(int argc, char** argv) {
    std::cout << "usage: " << argv[0] << "Nx Ny Nz Batch"<< std::endl;
    std::cout << "   Nx    " << "matrix dimension in x direction "<< std::endl;
    std::cout << "   Ny    " << "matrix dimension in y direction "<< std::endl;
    std::cout << "   Nz    " << "matrix dimension in z direction "<< std::endl;
    std::cout << "   Batch " << "number of matrices generated "<< std::endl;
}

int main(int argc, char** argv) {
    /* check input arguments */
    if (argc != 5) {
        usage(argc, argv);
        exit(1);
    }

    unsigned int Nx = argc[1];
    unsigned int Ny = argc[2];
    unsigned int Nz = argc[3];
    unsigned int Batch = argc[4];

    /* Randomly generate data */
    cufftComplex *data = cudaMalloc((void**) &data, sizeof(cufftComplex)*Nx*Ny*Nz*Batch);
    for (unsigned int i = 0, i < Nx*Ny*Nz) {
        data[i].x = rand() / (float)RAND_MAX;
        data[i].y = 0;
        
    }

}


cufftComplex gen_data_1d(unsigned int Nx, unsigned int Batch) {
    /* randomly generate 1D data */
    cufftComplex *data = cudaMalloc((void**) &data, sizeof(cufftComplex)* Nx * Batch)

}
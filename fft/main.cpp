#include <iostream>


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

    unsigned int Nx = argc[1]
    unsigned int Ny = argc[2]
    unsigned int Nz = argc[3]
    unsigned int Batch = argc[4]

    /* randomly generate data */ 
    if (Nz == 0 ) {
        if (Ny == 0) {
            /* generate 1D vectors */
            gen_data_1d()
        }
        /* generate 2D martrix
    }

}
#include "2dfft.hpp"

void usage(int argc, char **argv) {
    cout << "usage: " << argv[0] << " Nmax" << endl;
    cout << "Nx (int): Dimension of 2D input array." << endl;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        usage(argc, argv);
        exit(1);
    }
}
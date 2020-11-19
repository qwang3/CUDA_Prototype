import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    filename = "result.csv"
    data = np.genfromtxt(filename, delimiter=",")

    plt.plot(data[1:, 0], data[1:, 1], label='FFTW')
    plt.plot(data[1:, 0], data[1:, 2], label='cuFFT')
    plt.legend()
    print(np.max(data[1:, 2]))

    plt.xlabel("Input array size")
    plt.ylabel("Forward FFT time (ms)")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, np.max(data[1:, 0]))
    plt.ylim(1e-4, np.max(data[1:, 1]))
    plt.grid()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

file_name = 'build/result_64_test.csv'

def plot_overlap_result(data):

    # plt.plot(data[1:, 0], data[1:, 1], label='fftw', )
    plt.plot(data[1:, 0], data[1:, 2], label='cufft sequential', color='b')
    plt.plot(data[1:, 0], data[1:, 3], label='cufft overlap 1', color='r')
    plt.plot(data[1:, 0], data[1:, 4], label='cufft overlap 2', color='k')

    plt.xlim([0, np.max(data[1:, 0])])
    plt.ylim([0, np.max(data[1:, 2])])
    plt.grid(True)
    plt.xlabel('Input batch size (N of 64x64x64)')
    plt.ylabel('Computation time (s)')
    plt.legend()
    plt.title('Computation & Communication Overlap')

    plt.savefig('overlap_result_2.png', dpi=500)

def plot_single_result(data, index, data_label=None):
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    ax[0].plot(data[1:, 0], data[1:, index], label=data_label)
    grad = np.gradient(data[1:, index])
    ax[1].plot(data[1:, 0], grad, label=data_label)

    ax[0].set_xlim([0, np.max(data[1:, 0])])
    ax[0].set_ylim([0, np.max(data[1:, index])])
    ax[0].grid(True)
    ax[0].set_ylabel('Computation time (s)')
    ax[0].legend()
    ax[1].set_ylabel('Slope')
    ax[1].axhline(y=1, color='red', linestyle='-')
    ax[1].legend()

    fig.tight_layout()
    ax[0].title.set_text('Computation Time Vs Input Size')

    plt.savefig('result.png')


if __name__ == "__main__":
    data = np.genfromtxt(file_name, delimiter=',')
    plot_overlap_result(data)
    plot_single_result(data, 4, data_label="cufft")


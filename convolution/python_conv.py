import numpy as np
from scipy.signal import fftconvolve

if __name__ == "__main__":
    N = 6
    in_one = np.asarray(np.arange(N**3))
    in_one = np.reshape(in_one, (N ,N, N))
    
    res = np.fft.fft(in_one)
    print(res[0, 0, 0])

    


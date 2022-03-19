import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from mstamp_stamp import mstamp


def plot_data(data):
    n_dim = data.shape[0]
    plt.figure()
    for i in range(n_dim):
        plt.subplot(n_dim,1,i+1)
        plt.plot(data[i])
    plt.show()


if __name__ == "__main__":
    mat = scipy.io.loadmat('toy_data.mat')
    sub_len = mat["sub_len"].item()
    data = mat["data"].transpose(1,0)
    matrix_profile, profile_index = mstamp(data, sub_len)
    plot_data(data)
    plot_data(matrix_profile)
    plot_data(profile_index)



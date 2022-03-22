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
    data = mat["data"]
    matrix_profile, profile_index = mstamp(data.T, sub_len)

    plot_motifs(data, sub_len, matrix_profile, dimensionality=2)
    plt.savefig("images/toy_data_orig.png")

    n = data.shape[0]
    shifts = [-0.5, 0.8, -0.6]
    scales = [0.5, 2, 0.8]
    for k in range(3):
        data[n//2:,k] += shifts[k]
        data[n//2:,k] *= scales[k]

    matrix_profile, profile_index = mstamp(data.T, sub_len)
    plot_motifs(data, sub_len, matrix_profile, dimensionality=2)
    plt.savefig("images/toy_data_shifted_scaled.png")

    # plot_data(data)
    # plot_data(matrix_profile)
    # plot_data(profile_index)



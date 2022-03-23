"""
@author: Adrien Cances and Aurelien Pion
"""
import  matplotlib.pyplot as  plt
import numpy as np

def plot_motifs(matrix_profile, data, sub_len, dimensionality=1, profile=None):
    motif_at = matrix_profile[dimensionality - 1, :].argsort()[:2]
    
    nb_dims, length = data.shape
    plt.figure(figsize=(20, 10))
    if profile is not None:
        t_1_index = profile[:,motif_at[0]]
        dims = t_1_index
    else:
        dims = np.arange(nb_dims)
    nb_dims = len(dims)
    k = 0
    for i in dims:
        plt.subplot(nb_dims+1,1,k+1)
        plt.plot(data[i])
        plt.title(f"$T_{i+1}$")
        for m in motif_at:
            plt.plot(range(m,m+sub_len), data[i][m:m+sub_len], c='r')
        # plt.xlim((0, matrix_profile.shape[1]))
        plt.xlim((0, length))
        k += 1

    plt.subplot(nb_dims+1, 1, nb_dims+1)
    plt.title('{}-dimensional Matrix Profile'.format(dimensionality))
    plt.plot(matrix_profile[dimensionality-1, :])
    for m in motif_at:
        plt.axvline(m, c='r')
    # plt.xlim((0, matrix_profile.shape[1]))
    plt.xlim((0, length))
    plt.tight_layout()
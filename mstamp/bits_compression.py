import numpy as np

def Dnorm(T):
    """Normalize the time series T

    Parameters:
        T: np.array(m, dim_selected)

    Return:
        normalized sequence
    """
    m, dim = T.shape
    maxi = np.max(T, axis=0)
    mini = np.min(T, axis=0)
    b = int(T.nbytes/(m*dim))
    normalized = (T-mini[None,:])/(maxi[None,:] - mini[None,:])*(np.power(2, b) - 1) + 1
    return normalized.astype(int) + 1

def DL(T):
    """compute DL of T
    T must be normalized

    Parameters:
        T: np.array(m, dim_selected)

    Return:
        DL score of T
    """
    m, dim = T.shape
    return T.nbytes*8/dim

def RDL(t_c, t_h):
    """compute RDL of t_c according to t_h
    t_c and t_h must be normalized
    the bit number encoding b of t_c and t_h must be equal

    Parameters:
        t_c: np.array(m, dim_selected)
        t_h: np.array(m, dim_selected)

    Return:
        RDL score between t_c and t_h
    """
    diff = t_c - t_h
    gamma = np.sum(diff != 0)
    m, dim = t_c.shape
    b = int(t_c.nbytes/(m*dim))*8
    # b = int(t_c.nbytes/(m*dim))
    return gamma*(np.log2(m) + b)/dim

def bit(time_series, compressible_set_index, hypothesis_index, sub_len, unexplored_index=None):
    """Compute the bits required to store compressible_set_index according to hypothesis_index

    Parameters:
        time_series: list of index
        compressible_set_index: list of index
        hypothesis_index: list of index
        sub_len: int
        unexplored_index: list of index

    Return:
        number of bits necessary for compression using the article of Yeh,C.-C.M.,VanHerle,H.,andKeogh
    """
    compressible_set = []
    hypothesis_set = []
    for idx, prof in compressible_set_index:
        compressible_set.append(Dnorm(time_series[idx: idx + sub_len, prof]))
    for idx, prof in hypothesis_index:
        hypothesis_set.append(Dnorm(time_series[idx: idx + sub_len, prof]))
    h = len(hypothesis_set)
    res = 0
    for t_c in compressible_set:
        min_rdl = RDL(t_c, hypothesis_set[0])
        for i in range(1, h):
            rdl = RDL(t_c, hypothesis_set[i])
            if rdl <= min_rdl:
                min_rdl = rdl
        res += min_rdl
    for t_h in hypothesis_set:
        res += DL(t_h)
    if unexplored_index is not None:
        for t_u in unexplored_index:
            res += DL(t_u)
    return res

def MDL(matrix_profile, time_series, sub_len, profile):
    """mdl methods, compute the bits that are needed to store the
    two subsequences selected for each dimension

    Parameters:
        matrix_profil: np.array (n_dim, m)
        time_series: np.array (m, n_dim)
        sub_len: int
        profile: list of int (n_dim lists)

    Return:
        compression value for each dimension computed with RDL
    """
    dimension = matrix_profile.shape[0]
    bits = []
    for d in range(dimension):
        t_1, t_2 = matrix_profile[d, :].argsort()[:2]
        hypothesis_index = [(t_1, profile[d][:,t_1])]
        compressible_set_index = [(t_2, profile[d][:,t_2])]
        bits.append(bit(time_series, compressible_set_index, hypothesis_index, sub_len))
    return np.array(bits)

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def convert_df_f(data, quantile=0.08, scale_window=500):
    """ baseline correction"""

    baseline = (
        pd.DataFrame(data)
        .T.rolling(scale_window, center=True, min_periods=0)
        .quantile(quantile)
        .values.T
    )

    df_f = data / baseline -1
    #df_f[df_f<0] = 0

    return df_f

def denoise_sd(arr, axis=1, highbound=2, lowbound=0.5):
    # lowbound original setting is 0.5, highbound original is 2
    if axis == 0:
        arr = np.asarray(arr) - np.median(arr, axis=axis)
    else:
        arr = np.asarray(arr) - np.median(arr, axis=axis)[:, None]
    if len(arr.shape) == 1:
        arr = np.expand_dims(arr, axis=0)

    if axis == 0:
        arr = arr.T

    std = np.std(arr, axis=1)
    is_spike = arr > highbound * std[:, None]

    a, b = arr.shape
    for i in range(a):
        for j in range(b - 1):
            if not is_spike[i, j]:
                continue
            if is_spike[i, j + 1]:
                continue

            if arr[i, j + 1] > lowbound * std[i]:
                is_spike[i, j + 1] = True

    arr[~is_spike] = 0
    if axis == 0:
        arr = arr.T
    return arr


def find_transients(fc3, rois, prominence_thresh=0.1, amplitude=0.12, interval=10, width_min=5, width_max=100):

    transients_mat = np.zeros_like(fc3)

    for n in range(fc3.shape[0]):
        peaks, properties = find_peaks(fc3[n, :], prominence=prominence_thresh, height=amplitude, distance=interval,
                                       width=[width_min, width_max])
        for left_idx, right_idx in zip(properties['left_bases'], properties['right_bases']):
            transients_mat[n, left_idx:right_idx+1] = fc3[n, left_idx:right_idx+1]

    final_rois = np.where(np.sum(transients_mat, axis=1)>0)[0]

    return transients_mat[final_rois, :], rois[final_rois]


import numpy as np
from scipy.stats import zscore
from scipy.signal import savgol_filter
from src.motion_detection import segment_and_display, label_display, replace_seg_with_nan, add_seg_new_roi
from sklearn.decomposition import PCA
from src.roi_selection import SelectROI
from src.transient_detection import convert_df_f, denoise_sd, find_transients
import matplotlib.pyplot as plt

def smooth_data(raw: np.ndarray, order: int=1, window_size:int = 10):

    v = np.log(raw)
    s = savgol_filter(v, window_length=window_size, polyorder=order)
    s_reconstructed = np.exp(s)

    return s_reconstructed

def detect_motion(data: np.ndarray, rois: np.ndarray) -> np.ndarray:

    pca = PCA(n_components=1)

    z_data = zscore(data[rois, :], axis=1)
    z1 = pca.fit_transform(z_data.transpose())

    plt.plot(z1)
    plt.ion()
    plt.show(block=False)

    # Step 1: Ask user for the number of change points
    n_change_points = int(input("Enter the number of change points to detect (e.g. 4): "))

    # Step 2: Perform segmentation
    change_points = segment_and_display(z1, n_change_points)

    # Step 3: Ask the user if the result is good
    result_good = input("Is the segmentation result good? (yes/no): ").lower()

    while result_good != "yes":
        # If not good, ask for a different number of change points and rerun
        n_change_points = int(input("Enter a different number of change points: "))
        change_points = segment_and_display(z1, n_change_points)
        result_good = input("Is the segmentation result good? (yes/no): ").lower()

    # Step 4: If good, ask for chunks to replace with NaN
    seg_to_replace = input(
        "Enter the numbers of segments to replace with NaN, separated by spaces (e.g. 1 3), or press Enter to keep all: ")

    if seg_to_replace:
        seg_to_replace = list(map(int, seg_to_replace.split()))
        data_new = replace_seg_with_nan(data[rois, :], change_points, seg_to_replace)

    new_roi_to_add = input(
        "Enter the numbers of segments to add as new ROIs, separated by spaces to add multiple segments as the same FOV"
        " (e.g. 1 3) or iteratively as separate FOVs (e.g. 1), or press Enter to skip: ")

    while new_roi_to_add != "":
        seg_to_add = list(map(int, new_roi_to_add.split()))
        new = add_seg_new_roi(data[rois, :], change_points, seg_to_add)
        data_new = np.concatenate([data_new, new], axis=0)

        new_roi_to_add = input(
            "Enter the numbers of segments to add as new ROIs, separated by spaces to add multiple segments as the same FOV"
            " (e.g. 1 3) or iteratively as separate FOVs (e.g. 1), or press Enter to skip: ")

    # Final time series after replacing chunks with NaN
    print(f"Finished. Now data in shape: {data_new.shape}")
    return data_new


def handle_nan_frames(data, rois):
    nan_mask = np.isnan(data).any(axis=0)
    non_nan = data[:, ~nan_mask]

    df_f = convert_df_f(non_nan, quantile=0.08, scale_window=500)
    fc3 = denoise_sd(df_f)
    transients, selected_rois = find_transients(fc3, rois)

    combined = np.full((data.shape[0], data.shape[1]), np.nan)
    combined[:, ~nan_mask] = transients

    return combined, selected_rois
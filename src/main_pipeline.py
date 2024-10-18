# to avoid kmeans memory leak warning message, run this before importing kmeans
import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from scipy.stats import zscore
from scipy.signal import savgol_filter
from src.motion_detection import segment_and_display, label_display, replace_seg_with_nan, add_seg_new_roi
from sklearn.decomposition import PCA
from src.roi_selection import SelectROI
from src.transient_detection import remove_nan_frames, restore_nan_frames, detect_transients
from src.group_axon import GroupAxon
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
    plt.title('all ROIs PCA 1st component')
    plt.show()
    plt.pause(0.1)

    # Step 1: Ask user for the number of change points
    n_change_points = int(input("Enter the number of change points to detect (e.g. 4): "))

    # Step 2: Perform segmentation
    change_points = segment_and_display(z1, n_change_points)

    # Step 3: Ask the user if the result is good
    result_good = input("Is the segmentation result good? (yes/no): ").lower()

    while result_good != "yes":
        # If not good, ask for a different number of change points and rerun
        n_change_points = int(input("Enter a different number of change points: "))
        print(f"{n_change_points} changed points")
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
        print(f"added {seg_to_add} as new ROI")

        new_roi_to_add = input(
            "Enter the numbers of segments to add as new ROIs, separated by spaces to add multiple segments as the same FOV"
            " (e.g. 1 3) or iteratively as separate FOVs (e.g. 1), or press Enter to skip: ")

    # Final time series after replacing chunks with NaN
    print(f"Finished. Now data in shape: {data_new.shape}")
    return data_new, change_points


def validate_motion_static_channel(red, rois, change_points):

    pca = PCA(n_components=1)

    z_data = zscore(red[rois, :], axis=1)
    z_red = pca.fit_transform(z_data.transpose())

    label_display(z_red, change_points)


def _separate_fov(data):
    # Get the boolean mask for NaNs
    nan_mask = np.isnan(data)

    # Convert the NaN mask into a tuple of columns with NaNs for each row (hashable)
    nan_patterns = [tuple(row) for row in nan_mask]

    # Find the unique NaN patterns and their corresponding indices
    unique_patterns, fov = np.unique(nan_patterns, axis=0, return_inverse=True)

    # Separate the arrays based on unique patterns
    separated_arrays = []
    for pattern in range(len(unique_patterns)):
        separated_arrays.append(data[fov == pattern])

    return separated_arrays


def detect_transients_only(data, rois):

    data_fov_separated = _separate_fov(data)
    transients_only = []
    original_roi = []

    for data_fov in data_fov_separated:
        non_nan_data_fov, nan_mask = remove_nan_frames(data_fov)
        transients, selected_rois = detect_transients(non_nan_data_fov, rois)
        nan_data_fov = restore_nan_frames(transients, nan_mask, data_fov.shape)
        transients_only.append(nan_data_fov)
        original_roi.append(selected_rois)

    return np.concatenate(transients_only), original_roi


def detect_transients_axons(data, rois, grouping_class: GroupAxon):

    data_fov_separated = _separate_fov(data)
    axon_mat = []
    roi_map = []

    for data_fov in data_fov_separated:
        non_nan_data_fov, nan_mask = remove_nan_frames(data_fov)
        transients, selected_rois = detect_transients(non_nan_data_fov, rois)

        # group axons using GroupAxon class function
        print(f"Grouping axons using params: {grouping_class.get_params()}")
        combined_mat, group_map = grouping_class.combine_rois(transients, selected_rois)

        nan_data_fov = restore_nan_frames(combined_mat, nan_mask, (combined_mat.shape[0],data_fov.shape[1]))
        axon_mat.append(nan_data_fov)
        roi_map.append(group_map)

    return np.concatenate(axon_mat), roi_map
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

def segment_and_display(data, n_change_points, jump=5):
    # Perform bottom-up segmentation with n change points
    algo = rpt.Binseg(model="l2", jump=jump).fit(data)
    change_points = algo.predict(n_bkps=n_change_points)

    label_display(data, change_points)

    return change_points


def label_display(data, change_points):
    rpt.display(data, change_points)

    # Label each chunk
    start = 0
    text_y = np.nanmin(data)
    for i, cp in enumerate(change_points):
        plt.text((start + cp) / 2, text_y, f'{i + 1}', ha='center', fontsize=12)
        start = cp

    plt.show()
    plt.pause(0.1)


def replace_seg_with_nan(data, change_points, seg_to_replace):
    # Replace specified chunks with NaN
    new = np.copy(data)

    start_idx = 0
    for i, cp in enumerate(change_points):
        if i + 1 in seg_to_replace:
            new[:, start_idx:cp] = np.nan  # Replace values with NaN
        start_idx = cp

    return new


def add_seg_new_roi(data, change_points, seg_to_add):
    seg = np.setdiff1d(np.arange(len(change_points)+1), seg_to_add)
    new = replace_seg_with_nan(data, change_points, seg)

    return new

import numpy as np
from scipy.signal import periodogram
import matplotlib.pyplot as plt


class SelectROI:
    def __init__(self, transients_low = 0.03, transients_high = 0.13, drifts_low = 0.0001, drifts_high=0.01, fs=15.49,
                 norm_power_thresh = 0.3):
        self._transients_low = transients_low
        self._transients_high = transients_high
        self._drifts_low = drifts_low
        self._drifts_high = drifts_high
        self._fs = fs
        self._nyquist = self._fs/2
        self._norm_power_thresh = norm_power_thresh

    def calculate_norm_power(self, data: np.ndarray):
        n_roi, n_frames = np.shape(data)
        power_neuron = np.zeros((n_roi, int(np.floor(n_frames / 2) + 1)))
        for n in range(n_roi):
            f, Pxx = periodogram(data[n, :], fs=self._fs)
            power_neuron[n, :] = Pxx

        total_power = np.sum(power_neuron, axis=1)
        norm_power = power_neuron / total_power[:, np.newaxis]

        return norm_power, f

    def plot_norm_power(self, data: np.ndarray, good_rois: []):
        """ optional: finding frequency profile of calcium transients by plotting relative frequency bands power of
        good ROIs against population mean. recommended when using a new virus """

        norm_power, f = self.calculate_norm_power(data)
        mean_norm_power = np.mean(norm_power, axis=0)

        # plot example cells in time and frequency domains
        for n in range(len(good_rois)):
            fig, axs = plt.subplots(1, 2)

            axs[0].plot(data[good_rois[n], :])
            axs[0].set_title(f'Time domain cell {good_rois[n]}')

            axs[1].plot(f, norm_power[good_rois[n], :])
            axs[1].set_title(f'Freq domain cell {good_rois[n]}')
            axs[1].set_xlim([self._drifts_low, self._transients_high+self._transients_low])

            plt.tight_layout()
            plt.show()

        # plot mean normalized power of example rois vs. all rois
        plt.plot(f, mean_norm_power, color='black', label='mean normalized power: all rois included')
        mean_roi_power = np.mean(norm_power[good_rois, :], axis=0)
        plt.plot(f, mean_roi_power, color='green', label='mean normalized power: example rois only')
        plt.legend()
        plt.xlim([self._drifts_low, self._transients_high+self._transients_low])
        plt.title('normalized power')
        plt.show()

    def rank_rois(self, data: np.ndarray):

        norm_power, f = self.calculate_norm_power(data)

        ind_min = np.argmax(f > self._transients_low) - 1
        ind_max = np.argmax(f > self._transients_high) - 1
        ind_drift_min = np.argmax(f > self._drifts_low)

        lowf_power = np.zeros(data.shape[0])
        for n in range(data.shape[0]):
            lowf_power[n] = (np.trapz(norm_power[n, ind_min: ind_max], f[ind_min: ind_max]) /
                             np.trapz(norm_power[n, ind_drift_min::], f[ind_drift_min::]))

        rois = np.where(lowf_power > self._norm_power_thresh)[0]
        print(f'{len(rois)} rois selected using power > {self._norm_power_thresh}')

        plt.hist(lowf_power)
        plt.xlabel(f'low f ({self._transients_low} to {self._transients_high}) power distribution')
        plt.ylabel('roi count')
        plt.axvline(x=self._norm_power_thresh, color='r')
        plt.show()

        f_range = np.where((f > self._drifts_low) & (f < self._drifts_high))[0]
        drift_power = np.sum(np.log(norm_power[:, f_range]), axis=1)
        rois = rois[np.argsort(drift_power[rois])]

        return rois

    @property
    def drifts_low(self):
        return self._drifts_low

    @drifts_low.setter
    def drifts_low(self, drifts_low: float):
        self._drifts_low = drifts_low
        print(f"Drifts low set to: {self._drifts_low} Hz")

    @property
    def drifts_high(self):
        return self._drifts_high

    @drifts_high.setter
    def drifts_high(self, drifts_high: float):
        self._drifts_high = drifts_high
        print(f"Drifts high set to: {self._drifts_high} Hz")

    @property
    def transients_low(self):
        return self._transients_low

    @transients_low.setter
    def transients_low(self, transients_low: float):
        self._transients_low = transients_low
        print(f"Transients low set to: {self._transients_low} Hz")

    @property
    def transients_high(self):
        return self._transients_high

    @transients_high.setter
    def transients_high(self, transients_high: float):
        self._transients_high = transients_high
        print(f"Transients high set to: {self._transients_high} Hz")

    @property
    def norm_power_thresh(self):
        return self._norm_power_thresh

    @norm_power_thresh.setter
    def norm_power_thresh(self, norm_power_thresh: float):
        self._norm_power_thresh = norm_power_thresh
        print(f"Norm power threshold set to {self._norm_power_thresh}")
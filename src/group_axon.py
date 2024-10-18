import numpy as np
import json
import os
from sklearn.decomposition import PCA
os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._traversal import connected_components
from sklearn.preprocessing import StandardScaler


class GroupAxon:
    def __init__(self, method: str, min_corr: float = 0.4, corr_threshold: float = None,
                 cluster_early_stop: float = None, baseline_quantile: float=0.08):
        """
        Initialize the SampleMergerPipeline with the method and threshold.
        Once the method and threshold are set, they cannot be changed for subsequent datasets.

        :param method: Method to combine samples. Must be 'correlation', 'hierarchical', or 'kmeans'.
        :param min_corr: A step to reduce computation cost. A ROI has to have a correlation value larger than min_corr
        to be considered to be included in a group. Otherwise excluded from the clustering step.
        :param corr_threshold: Correlation threshold to use if method is 'correlation'. Should be between 0 and 1.
        :param cluster_early_stop: A step to reduce computation cost if method is 'hierarchical' or 'kmeans'. Highly
        recommend to use cluster_early_stop especially when using 'kmeans'. Should be between 0 and 1. Out of all # of
        ROIs (e.g. 200), if silhouette score is not improving after cluster_early_stop (e.g. 0.1*200 = 20 tries), stop
        clustering and use the best in history as final result.
        :param baseline_quantile: When merging ROIs from the same group, use bottom quantile as baseline. Should be >0
        and <<0.5.
        """

        self.method = method
        self.min_corr = min_corr
        self.corr_threshold = corr_threshold
        self.cluster_early_stop = cluster_early_stop
        self.baseline_quantile = baseline_quantile
        self._validate_inputs()

    def _validate_inputs(self):

        if self.method not in ['correlation', 'hierarchical', 'kmeans']:
            raise ValueError("Invalid method. Choose 'correlation', 'hierarchical', or 'kmeans'.")

        if self.min_corr < 0 or self.min_corr > 1:
            raise ValueError("Invalid min_corr. Must be between 0 and 1.")

        if self.method == 'correlation' and (self.corr_threshold is None or not (0 <= self.corr_threshold <= 1)):
            raise ValueError("For 'correlation' method, a valid threshold between 0 and 1 must be provided.")

        if (self.method == 'hierarchical' or self.method == 'kmeans') and (self.cluster_early_stop < 0 or self.cluster_early_stop > 1):
            raise ValueError("For 'hierarchical' or 'kmeans' method, a valid threshold between 0 and 1 must be provided.")

    def get_params(self):
        return {'method': self.method, 'min_corr': self.min_corr, 'corr_threshold': self.corr_threshold,
                'cluster_early_stop': self.cluster_early_stop, 'baseline_quantile': self.baseline_quantile}

    def combine_rois(self, data: np.ndarray, rois: np.ndarray, fast: bool=True):
        """
        Combine samples in the given dataset using the predefined method and threshold. First remove singletons (defined
        by max correlation with other ROIs smaller than min_corr), then use predefined method to group rest of ROIs.

        :param data: 2D NumPy array of shape (n_samples, n_timepoints).
        :return: Tuple containing the dimension-reduced data and a dictionary mapping groups to original samples.
        """

        active_rois = np.where(np.nansum(data, axis=1)>0)[0]
        data = data[active_rois, :]
        rois = rois[active_rois]
        corr = np.corrcoef(data)

        if fast == True:
            corr, grouping_rois = self._reduce_clustering_size(corr)
            rois = rois[grouping_rois]

        # perform grouping using predefined method and threshold
        if self.method == 'correlation':
            combined_mat, group_map = self._group_axons_by_correlation(data, corr, rois)
        elif self.method == 'hierarchical':
            combined_mat, group_map = self._group_axons_by_hierarchical_clustering(data, corr, rois)
        elif self.method == 'kmeans':
            combined_mat, group_map = self._group_axons_by_kmeans(data, corr, rois)

        return combined_mat, group_map

    def _reduce_clustering_size(self, corr):

        # find singletons by using min_corr as cutoff on pairwise correlation between ROIs
        np.fill_diagonal(corr, np.nan)
        max_corr = np.nanmax(corr, axis=0)

        grouping_rois = np.where(max_corr > self.min_corr)[0]
        singleton_rois = np.where((max_corr > 0) & (max_corr < self.min_corr))[0]

        grouping_corr = corr[grouping_rois, :]
        grouping_corr = grouping_corr[:, grouping_rois]
        np.fill_diagonal(grouping_corr, 1)
        n_max_cluster = len(grouping_rois)
        print(f'# potential non-singleton ROIs: {n_max_cluster}, # singleton ROIs: {len(singleton_rois)}')
        plt.hist(max_corr)
        plt.title('max corr per ROI')
        plt.axvline(x=self.min_corr, color='r')
        plt.show()

        return grouping_corr, grouping_rois

    def _group_axons_by_correlation(self, data, corr, rois):

        n_comp, group_ids = connected_components(csr_matrix(corr > self.corr_threshold), directed=False, return_labels=True)
        combined_mat, group_map = self._merge_within_group_roi(data, n_comp, group_ids, rois)

        return combined_mat, group_map

    def _group_axons_by_hierarchical_clustering(self, data: np.ndarray, corr: np.ndarray, rois: np.ndarray):
        """
        Perform hierarchical clustering and find the optimal number of clusters using silhouette score.
        :param data: 2D NumPy array of sample data.
        :param n_samples: Number of samples in the dataset.
        :return: Cluster labels for each sample.
        """
        n_cluster = np.arange(2, len(rois))
        silhouette_all = np.zeros(len(n_cluster))
        cluster_results = np.zeros((len(n_cluster), len(rois)))
        stopping_thresh = int(np.ceil(self.cluster_early_stop * len(n_cluster)))

        scaled_corr = StandardScaler().fit_transform(corr)

        best_n_clusters = -1
        best_silhouette = -1

        for k in range(len(n_cluster)):  # Test different cluster sizes
            clustering = AgglomerativeClustering(n_clusters=n_cluster[k])
            cluster_labels = clustering.fit_predict(scaled_corr)
            cluster_results[k, :] = cluster_labels
            silhouette_all[k] = np.round(silhouette_score(scaled_corr, cluster_labels), 3)

            if silhouette_all[k] > best_silhouette:
                best_silhouette = silhouette_all[k]
                best_n_clusters = n_cluster[k]
                dropping_count = 0
                print(f"cluster {n_cluster[k]} Silhouette score: {silhouette_all[k]}, new best")
            else:
                dropping_count += 1
                print(f"cluster {n_cluster[k]} Silhouette score: {silhouette_all[k]}, counting {dropping_count}")

            if dropping_count >= stopping_thresh:
                print(f"best K: {best_n_clusters}, best Silhouette score: {best_silhouette}")
                break

        plt.plot(n_cluster[:k], silhouette_all[:k], label='silhouette')
        plt.scatter(best_n_clusters, best_silhouette, color='red', marker='x')
        plt.title(f"Hierarchical clustering Silhouette score")
        plt.xlabel('# clusters')
        plt.ylabel('Silhouette score')
        plt.show()

        combined_mat, group_map = self._merge_within_group_roi(data, best_n_clusters, cluster_results[best_n_clusters],
                                                               rois)

        return combined_mat, group_map

    def _group_axons_by_kmeans(self, data: np.ndarray, corr: np.ndarray, rois: np.ndarray):
        """
        Perform hierarchical clustering and find the optimal number of clusters using silhouette score.
        :param data: 2D NumPy array of sample data.
        :param n_samples: Number of samples in the dataset.
        :return: Cluster labels for each sample.
        """
        n_cluster = np.arange(2, len(rois))
        silhouette_all = np.zeros(len(n_cluster))
        cluster_results = np.zeros((len(n_cluster), len(rois)))
        stopping_thresh = int(np.ceil(self.cluster_early_stop * len(n_cluster)))

        scaled_corr = StandardScaler().fit_transform(corr)

        best_n_clusters = -1
        best_silhouette = -1

        for k in range(len(n_cluster)):
            clustering = KMeans(n_clusters=n_cluster[k])
            cluster_labels = clustering.fit_predict(scaled_corr)
            cluster_results[k, :] = cluster_labels
            silhouette_all[k] = np.round(silhouette_score(scaled_corr, cluster_labels), 3)

            if silhouette_all[k] > best_silhouette:
                best_silhouette = silhouette_all[k]
                best_n_clusters = n_cluster[k]
                dropping_count = 0
                print(f"cluster {n_cluster[k]} Silhouette score: {silhouette_all[k]}, new best")
            else:
                dropping_count += 1
                print(f"cluster {n_cluster[k]} Silhouette score: {silhouette_all[k]}, counting {dropping_count}")

            if dropping_count >= stopping_thresh:
                print(f"best K: {best_n_clusters}, best Silhouette score: {best_silhouette}")
                break

        plt.plot(n_cluster[:k], silhouette_all[:k],  label='silhouette')
        plt.scatter(best_n_clusters, best_silhouette, color='red', marker='x')
        plt.title(f"Kmeans clustering Silhouette score")
        plt.xlabel('# clusters')
        plt.ylabel('Silhouette score')
        plt.show()

        combined_mat, group_map = self._merge_within_group_roi(data, best_n_clusters, cluster_results[best_n_clusters],
                                                               rois)

        return combined_mat, group_map

    def _merge_within_group_roi(self, data, n_comp, group_ids, rois):

        grouped_components = []
        group_map = {}
        pca = PCA(n_components=1)

        for gid in range(n_comp):
            idx = np.where(group_ids == gid)[0]
            grouping_ind = list(idx)
            group_map[gid] = rois[grouping_ind]
            roi_group = data[idx].T
            if roi_group.shape[-1] == 1:
                result = roi_group
            else:
                result = pca.fit_transform(roi_group)
                result /= pca.components_.sum(1)
            result = result - np.quantile(result, self.baseline_quantile)
            grouped_components.append(np.squeeze(result))

        return np.vstack(grouped_components), group_map
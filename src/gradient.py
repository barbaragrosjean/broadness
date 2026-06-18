import numpy as np
import os
import pandas as pd
import scipy
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import plotting

from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score

from src.utils import mnicorr2niigz, OUT_PATH, mni_coords


@dataclass
class ClusteringResult:
    labels: np.ndarray
    components:np.ndarray
    method: str

    def __post_init__(self):
        self.path_to_save =  OUT_PATH + '/gradient'
        os.makedirs(self.path_to_save, exist_ok=True)

    def plot_clusters_brain(self, save_fig= False, save_niigz=False, MNI_path=mni_coords):
        cluster_labels = self.labels
        set1_colors = plt.cm.Set2(np.linspace(0, 1, len(np.unique(cluster_labels))))
        img = mnicorr2niigz(cluster_labels + 1, self.path_to_save + f'/cluster_{self.method}', save =save_niigz)
        fig= plotting.plot_glass_brain(img, display_mode='lyrz', cmap=ListedColormap(np.vstack([[1, 1, 1, 1], set1_colors])), title=f'{self.method} Clusters')
        
        if save_fig : 
            fig.savefig(self.path_to_save + f'/cluster_{self.method}.png')
        return fig

def cluster_number(compo:np.ndarray, range_n_clusters = np.arange(2, 12, 1), summary=True) : 
    """
    Determine the optimal number of clusters for K-means using the silhouette score.

    Parameters
    ----------
    compo : np.ndarray
        Array of shape (n_samples, n_features) representing the components to cluster.
    range_n_clusters : np.ndarray, optional
        Array of integers specifying the candidate number of clusters (default = np.arange(2, 12)).
    summary : bool, optional
        If True, prints progress and the optimal number of clusters (default = True).

    Returns
    -------
    ncl_best : int
        The number of clusters that maximizes the silhouette score.

    Raises
    ------
    TypeError
        If `compo` is not a 2D numpy array or `range_n_clusters` is not iterable.
    ValueError
        If `range_n_clusters` contains invalid values or `compo` has insufficient samples.
    """
    # -------------------
    # Input validation
    # -------------------
    if not isinstance(compo, np.ndarray) or compo.ndim != 2:
        raise TypeError("compo must be a 2D numpy array (n_samples, n_features).")
    if not hasattr(range_n_clusters, "__iter__"):
        raise TypeError("range_n_clusters must be an iterable of integers.")
    if not np.all(range_n_clusters >= 1):
        raise ValueError("All values in range_n_clusters must be >= 1.")
    if not isinstance(summary, bool):
        raise TypeError("summary must be a boolean.")

    if summary :
        print('Selecting the right number of components for K-means ...')
    # Find the right number of cluster
    silhouette_avg = []

    for n_clusters in range_n_clusters : 
        kmean = KMeans(n_clusters)
        kmean.fit(compo)
        cluster_labels = kmean.labels_
        silhouette_avg.append(silhouette_score(compo, cluster_labels))

    ncl_best = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]

    if summary :
        print(f'With {ncl_best} cluster(s) we have a silhouette score of {max(silhouette_avg)}')

    return ncl_best


def clustering(compo: np.ndarray, method_cls = 'kmeans', summary=True, thr_compo=True):
    """
    Perform dimensionality reduction (PCA/ICA) followed by clustering 
    (KMeans, DBSCAN, SpectralClustering) on neuroimaging data.

    Parameters
    ----------
    compo : np.array

    method_cls : str, optional
        Clustering method: 'kmeans', 'DBSCAN', or 'SpectralClustering' (default='kmeans').
    summary : bool, optional
        If True, prints information and plots results (default=False).
    method_decom : str, optional
        Decomposition method: 'pca' or 'ica' (default='pca').
    MNI_path : str, optional
        Path to MNI coordinates .mat file (default='data/MNI152_8mm_coord_dyi.mat').
    save_niigz : bool, optional
        If True, saves clustering results as NIfTI image (default=False).
    thr_compo : bool, optional
        If True, thresholds each component by mean+std (default=True).

    Returns
    -------
    cluster_labels : np.ndarray
        Array of cluster labels for each voxel/source.

    Raises
    ------
    TypeError
        If input types are invalid.
    ValueError
        If method_decom or method_cls is unsupported.
    """

    # -------------------
    # Input validation
    # -------------------
    if method_cls not in ['kmeans', 'DBSCAN', 'SpectralClustering']:
        raise ValueError("method_cls must be 'kmeans', 'DBSCAN', or 'SpectralClustering'.")
    if not isinstance(summary, bool):
        raise TypeError("summary must be a boolean.")
    
    if summary:
        print('--------------------------------------------')
        print(f'{method_cls} on {compo.shape[0]} PCs')
        print('--------------------------------------------')

    
    if method_cls == 'kmeans' : 
        ncl_best = cluster_number(compo.T, summary=summary)
        kmean = KMeans(ncl_best).fit(compo.T)
        cluster_labels = kmean.labels_

    if method_cls == 'DBSCAN' : 
        cls = DBSCAN(eps=0.5, min_samples=5).fit(compo.T)
        cluster_labels = cls.components_

    if method_cls == 'SpectralClustering' : 
        cls = SpectralClustering().fit(compo.T)
        cluster_labels = cls.labels_

    return ClusteringResult(cluster_labels,compo.T, method_cls)
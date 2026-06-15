import pandas as pd
import numpy as np
import os
import mat73
import scipy

from sklearn.decomposition import PCA, FastICA

import nibabel as nib
import matplotlib.pyplot as plt

    
def compute_significant_pcs(data, variance, permutations_num=100, randomization=1, summary=False):
    """
    Compute statistically significant principal components (PCs) via randomization tests = Monte Carlo simulation (MCS).

    This function compares the variance explained by the observed PCs with the variance
    distribution obtained from randomized versions of the data. PCs that explain more
    variance than the maximum variance observed in the randomized data are considered
    significant.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (features, time) or (voxels, time).
    variance : np.ndarray
        Explained variance ratio of the observed PCs, from PCA.
    permutations_num : int, optional
        Number of random permutations to perform (default = 100).
        If <= 0, no permutation testing is done and all PCs are returned.
    randomization : {1, 2, 3}, optional
        Randomization scheme:
        - 1 : Shuffle along time dimension
        - 2 : Shuffle along space dimension
        - 3 : Shuffle along both space and time
        Default = 1.
    summary : bool, optional
        If True, prints details about the randomization and significant PCs.

    Returns
    -------
    PCs : np.ndarray
        Indices of significant PCs.

    Raises
    ------
    TypeError
        If inputs are of incorrect type.
    ValueError
        If `permutations_num` < 0, or `randomization` is not in {1, 2, 3}.
    """
    # ____________
    # Check inputs
    # ____________
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray.")
    if data.ndim != 2:
        raise ValueError("data must be a 2D array (features × time).")
    if not isinstance(variance, np.ndarray):
        raise TypeError("variance must be a numpy.ndarray.")
    if variance.ndim != 1:
        raise ValueError("variance must be a 1D array of explained variance ratios.")
    if not isinstance(permutations_num, int):
        raise TypeError("permutations_num must be an integer.")
    if permutations_num < 0:
        raise ValueError("permutations_num must be non-negative.")
    if randomization not in {1, 2, 3}:
        raise ValueError("randomization must be 1 (time), 2 (space), or 3 (both).")
    if not isinstance(summary, bool):
        raise TypeError("summary must be a boolean (True/False).")
    
    # ______________
    # Main Function
    # ______________

    if permutations_num <= 0:
        if summary :
            print('No permutation computed.')
        # no randomization requested → keep all PCs
        return np.arange(len(variance))
    if summary :
        print("Computing PCA on randomized data...")
        if randomization == 1 :
            print('Randomization on Time dimention')
        elif randomization == 2 : 
            print('Randomization on Space dimention')
        elif randomization == 3 : 
            print('Randomization on Space and Time dimention')

    max_variances = []
    all_variances = []

    for permi in range(permutations_num):
        if randomization == 1:  # Time
            idx = np.apply_along_axis(np.random.permutation, 1, data)
            data_reshaped = idx

        elif randomization == 2:  # Space
            idx = np.apply_along_axis(np.random.permutation, 0, data)
            data_reshaped = idx

        elif randomization == 3:  # Both
            flat = data.flatten()
            np.random.shuffle(flat)
            data_reshaped = flat.reshape(data.shape)

        # PCA on randomized data (transpose to match MATLAB’s convention)
        pca = PCA()
        pca.fit(data_reshaped.T)
        var_rand = pca.explained_variance_ratio_

        all_variances.append(var_rand)
        max_variances.append(np.max(var_rand))

    variance_randomized_mean = np.mean(np.column_stack(all_variances), axis=1)
    max_variance = np.max(max_variances)
    PCs = np.where(variance > max_variance)[0]

    if summary : 
        if PCs.size == 0 :
            print("No PCs (brain networks) survived MCS")
        else:
            print("Variance explained by significant PCs (brain networks):")
            print(variance[PCs])

    return PCs

def load_data(file_path) :
    """
    Load neuroimaging data from a .mat or .csv file.

    The function expects the data to represent a 3D array in the format:
    (sources × time points × conditions).

    Parameters
    ----------
    file_path : str
        Path to the input file. Must be either:
        - `.mat` file (MATLAB v7.3 format, containing key 'data'), or
        - `.csv` file (rows = sources × time, columns = conditions or similar).

    Returns
    -------
    data : np.ndarray
        3D numpy array of shape (n_sources, n_timepoints, n_conditions).

    Raises
    ------
    TypeError
        If `file_path` is not a string.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not supported or if the loaded data
        does not have the expected 3D shape.
    KeyError
        If the `.mat` file does not contain the key 'data'.
    """
    # -------------------
    # Input validation
    # -------------------
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check the file formate 
    if file_path[-3:] == 'mat':
        data = mat73.loadmat(file_path)
        data = data['data']

    elif file_path[:-3] == 'csv':
        data = pd.read_csv(file_path).values
    else : 
        print('Give csv or mat data')
        return 0
    
    # check the type of data : source, time point, condition
    if len(data.shape) != 3 : 
        print('Give proper data : (source, time, conditions)')
        return 0
    
    return data

def data_overview(data, conditions, method ='pca', nb_compo_to_try=10, thr_pca = 0.9, summary = True) :
    """
    Provide an overview of the dataset and estimate the number of components 
    to retain using PCA (or ICA placeholder).

    Parameters
    ----------
    file_path : str
        Path to the input data file (.mat or .csv). Must be loadable by `get_data`.
    conditions : dict
        Dictionary mapping condition names (keys) to trial indices (values).
        Example: {"rest": [0, 1, 2], "task": [3, 4, 5]}.
    method : str, optional
        Decomposition method to use. Currently supports:
        - 'pca' : Principal Component Analysis
        - 'ica' : Independent Component Analysis (not implemented yet).
        Default = 'pca'.
    nb_compo_to_try : int, optional
        Maximum number of components to try when computing PCA (default = 10).
    thr_pca : float, optional
        Variance threshold (between 0 and 1). The smallest number of components
        explaining at least this cumulative variance will be selected. Default = 0.8.
    summary : bool, optional
        If True, prints dataset info and displays variance plots (default = True).

    Returns
    -------
    nb_compo : int
        Number of components that meet the cumulative variance threshold.

    Raises
    ------
    TypeError
        If `file_path` is not a string, `conditions` is not a dict,
        or if `nb_compo_to_try` is not an int, `thr_pca` not a float,
        or `summary` not a bool.
    ValueError
        If `thr_pca` is not between 0 and 1, or if method is unsupported,
        or if no components reach the threshold.
    """
    # -------------------
    # Input validation
    # -------------------
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be an array.")
    if not isinstance(conditions, dict):
        raise TypeError("conditions must be a dictionary of {name: indices}.")
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if method not in ["pca", "ica"]:
        raise ValueError("method must be 'pca' or 'ica'.")
    if not isinstance(nb_compo_to_try, int) or nb_compo_to_try <= 0:
        raise TypeError("nb_compo_to_try must be a positive integer.")
    if not isinstance(thr_pca, float) or not (0 < thr_pca < 1):
        raise ValueError("thr_pca must be a float between 0 and 1.")
    if not isinstance(summary, bool):
        raise TypeError("summary must be a boolean.")
    
    # print overview and PCA overview if true 
    if summary : 
        print(f'Your data are: {data.shape[0]} sources, over {data.shape[1]} time points, {data.shape[2]} conditions: {list(conditions.keys())}')
        print('--------------------------------------------')
        print('Computing PCA on the mean over conditions :')
        print('--------------------------------------------')
    X_mean = data.mean(2)

    if method == 'pca' : 
        pca = PCA(n_components=nb_compo_to_try)
        pca.fit(X_mean.T)
        cumulative_variances = np.cumsum(pca.explained_variance_ratio_)
        nb_compo = np.where(cumulative_variances > thr_pca)[0][0] + 1

        if summary : 
            plt.plot(np.arange(nb_compo_to_try)+1, cumulative_variances)
            plt.grid()
            plt.plot([1, nb_compo_to_try], [thr_pca, thr_pca], c = 'r')
            plt.plot([nb_compo, nb_compo], [cumulative_variances[0], 1], c = 'r')
            plt.xticks(np.arange(nb_compo_to_try)+1)
            plt.xlabel('Componant Numbers')
            plt.ylabel('Cumulative Variance')
            plt.title('PCA cumulative explained variance')
            plt.show()

            print(f'{nb_compo} componant(s) account for {cumulative_variances[nb_compo-1] *100:.2f} variance explained:')
            for c in range(nb_compo): 
                print(f'Componant {c+1} : {pca.explained_variance_ratio_[c] *100:.2f} %')
            print(f'{nb_compo} componant(s) will be used by default for the following analysis. Please change manually if needed.')

    if method == 'ica' : 
        # TODO 

        print('not yet done')
    
    return int(nb_compo)

def mnicorr2niigz(compo, label_out='', save=False, path_glasser = "template/glasser360MNI.nii.gz", path_MNI = 'template/MNI152_8mm_coord_dyi.mat') :
    """
    Map component values to a volumetric NIfTI image using the Glasser atlas and MNI coordinates.

    Parameters
    ----------
    compo : np.ndarray
        1D array of component values, length = number of MNI coordinates (sources).
    label_out : str
        Output label for the NIfTI file. The file will be saved as "figure/{label_out}.nii.gz".
    save : bool, optional
        If True, saves the NIfTI image to disk (default = True).
    path_glasser : str, optional
        Path to the Glasser volumetric atlas NIfTI file (default = "data/Glasser_MNI.nii.gz").
    path_MNI : str, optional
        Path to the MNI coordinates `.mat` file (default = "data/MNI152_8mm_coord_dyi.mat").

    Returns
    -------
    new_img : nib.Nifti1Image
        NIfTI image with the component mapped onto the Glasser atlas.

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If array dimensions are incompatible or if lengths mismatch.
    FileNotFoundError
        If `path_glasser` or `path_MNI` does not exist.
    KeyError
        If the `.mat` file does not contain the key 'MNI8'.
    """

    # -------------------
    # Input validation
    # -------------------
    if not isinstance(compo, np.ndarray):
        raise TypeError("compo must be a numpy.ndarray.")
    if compo.ndim != 1:
        raise ValueError("compo must be a 1D array of component values.")

    if not isinstance(label_out, str):
        raise TypeError("label_out must be a string.")

    if not isinstance(save, bool):
        raise TypeError("save must be a boolean.")

    if not isinstance(path_glasser, str):
        raise TypeError("path_glasser must be a string.")
    if not os.path.exists(path_glasser):
        raise FileNotFoundError(f"Glasser atlas file not found: {path_glasser}")

    if not isinstance(path_MNI, str):
        raise TypeError("path_MNI must be a string.")
    if not os.path.exists(path_MNI):
        raise FileNotFoundError(f"MNI file not found: {path_MNI}")
    # Load Glasser volumetric atlas
    atlas_img = nib.load(path_glasser)
    atlas_data = atlas_img.get_fdata()  
    affine = atlas_img.affine
    glasser_signal = np.zeros_like(atlas_data) # init

    # Load MNI
    MNI_dcit = scipy.io.loadmat(path_MNI)
    mni_coords = MNI_dcit['MNI8'] 

    # Convert MNI coordinates to voxel indices
    vox_coords = nib.affines.apply_affine(np.linalg.inv(affine), mni_coords)
    vox_coords = np.round(vox_coords).astype(int)

    for i, coord in enumerate(vox_coords):
        x, y, z = coord
        parcel_label = atlas_data[x, y, z]  
        if parcel_label > 0:  # Ignore background
            glasser_signal[atlas_data == parcel_label] = compo[i]

    new_img = nib.Nifti1Image(glasser_signal, affine)

    if save ==True : 
        nib.save(new_img,f"outs/brain_networks/{label_out}.nii.gz")

    return new_img
# network_estimation.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import numpy as np
import pandas as pd
import scipy

import matplotlib.colors as cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nilearn import plotting

from sklearn.decomposition import PCA, FastICA

from utils import mnicorr2niigz
from config import OUT_PATH


@dataclass
class NetworkResult:
    components: np.ndarray
    explained_variance: np.ndarray
    timecourses: Dict[str, pd.DataFrame]
    reconstructed_activity: Dict[str, np.ndarray]
    conditions: Dict[str, int]

    @property
    def n_components(self) -> int:
        return self.components.shape[0]
    
    def get_nifti_component(self, n_compo, save=False, path_glasser: str = "template/glasser360MNI.nii.gz", path_MNI :str = 'template/MNI152_8mm_coord_dyi.mat'): 
        img = mnicorr2niigz(self.components[n_compo-1, :], label_out='brain_networks/component' + str(n_compo) , save=save, path_glasser=path_glasser,path_MNI=path_MNI )
        return img

    def plot_components_interactive(self, n_compo: int, path_glasser: str = "template/glasser360MNI.nii.gz", path_MNI :str = 'template/MNI152_8mm_coord_dyi.mat') :
        """
        Plot spatial components (brain networks) in 3D space using MNI coordinates.
        """

        # -------------------
        # Input validation
        # -------------------
        if not isinstance(n_compo, int): raise TypeError(f"nb_compo must be an integer between 1 to {self.components.shape[0] +1}")
        if n_compo <= 0: raise ValueError("nb_compo must be a positive integer.")
        if n_compo > self.components.shape[0]: raise ValueError(f"nb_compo cannot exceed the number of components ({self.components.shape[0]}).")
        if not isinstance(path_MNI, str): raise TypeError("path_MNI must be a string.")
        if not os.path.exists(path_MNI): raise FileNotFoundError(f"MNI file not found: {path_MNI}")
        
        # -------------------
        # Plot
        # -------------------
        img = self.get_nifti_component(n_compo, save=False,path_MNI=path_MNI, path_glasser=path_glasser)
        view = plotting.view_img_on_surf(img)

        return view

    def plot_components(self, n_compo: int, path_glasser: str = "template/glasser360MNI.nii.gz", path_MNI :str = 'template/MNI152_8mm_coord_dyi.mat', save=False) :
        """
        Plot spatial components (brain networks) in 3D space using MNI coordinates.
        """
        # -------------------
        # Input validation
        # -------------------
        if not isinstance(n_compo, int): raise TypeError(f"nb_compo must be an integer between 1 to {self.components.shape[0] +1}")
        if n_compo <= 0: raise ValueError("nb_compo must be a positive integer.")
        if n_compo > self.components.shape[0]: raise ValueError(f"nb_compo cannot exceed the number of components ({self.components.shape[0]}).")
        if not isinstance(path_MNI, str): raise TypeError("path_MNI must be a string.")
        if not os.path.exists(path_MNI): raise FileNotFoundError(f"MNI file not found: {path_MNI}")
        
        # -------------------
        # Plot
        # -------------------
        img = self.get_nifti_component(n_compo, save=False,path_MNI=path_MNI, path_glasser=path_glasser)
        img_static = plotting.plot_img_on_surf(img,
                                    hemispheres=["left", "right"], 
                                    cmap='seismic', 
                                    bg_on_data=True,
                                    surf_mesh="fsaverage5",
                                    colorbar=False,
                                    title=f"component {n_compo}")
        if save : 
            img_static[0].savefig(OUT_PATH + f'/brain_networks/components{n_compo}.png')

        return img_static

    
    def plot_timecourses(self, groupby: str, condition_to_plot = 'all', time_course_to_plot= 'all', save=False): # todo or one of the self.conditions
        if time_course_to_plot == 'all':
            time_course_to_plot = [j+1 for j in range(self.components.shape[0])]
            
        if condition_to_plot == 'all' : 
            condition_to_plot= list(self.conditions.keys())

        # 1 condi all the components ts
        if groupby == 'conditions' : 
            fig, ax = plt.subplots(1, len(condition_to_plot), figsize=(len(condition_to_plot)*5, 3))

            for j, c_name in enumerate(condition_to_plot):
                if len(condition_to_plot) == 1 : 
                    the_ax = ax
                else : 
                    the_ax=ax[j-1]

                the_ax.set_title(c_name)
                for i in time_course_to_plot : 
                    the_ax.plot(self.timecourses[c_name].loc[:, f'Net{i}_{c_name}'], label=f'Net_{i}')
                the_ax.grid()

        # all condition 1 ts per compo 
        if groupby == 'components' : 
            fig, ax = plt.subplots(1, len(time_course_to_plot), figsize=(len(time_course_to_plot)*5, 3))

            for j in time_course_to_plot:
                for i in condition_to_plot : 
                    if len(time_course_to_plot) == 1 : 
                        the_ax = ax
                    else : 
                        the_ax=ax[j-1]
                    the_ax.set_title('Net' + str(j))
                    the_ax.plot(self.timecourses[i].loc[:, f'Net{j}_{i}'], label=i)
                the_ax.grid()
        
        the_ax.legend()
        label= 'Net'
        for tc in time_course_to_plot :
            label= label.join(str(tc))

        for c in condition_to_plot :
            label=label.join('_' + c)
        

        if save : 
            fig.savefig(OUT_PATH + f'/brain_networks/time_courses_by_{groupby}_{label}.png')
    
    
    def plot_variance(self, save=False) : 
        fig, ax=plt.subplots(figsize=(8, 3))
        ax.plot(['PC' + str(i+1) for i in range(len(self.explained_variance))], self.explained_variance*100, c='k')
        ax.set_ylabel('Explained variance (%)')
        ax.grid()
        if save:
            fig.savefig(OUT_PATH + f'/brain_networks/expl_var.png')

class NetworkEstimator:
    def __init__(self,n_components: int,conditions: dict, method: str = "pca"):
        self.n_components = n_components
        self.method = method
        self.conditions = conditions

        self.model_ = None
        self.result_ = None

    def _validate_inputs(self):
        if self.n_components <= 0: raise ValueError("n_components must be positive.")
        # add compo < dim TODO
        if self.method not in ["pca", "ica"]: raise ValueError( "method must be 'pca' or 'ica'.")

    def fit(self, data: np.ndarray):

        self._validate_inputs()

        # If all subject have been passed we take the mean
        if data.ndim == 4: data = data.mean(axis=3)

        # Mean over the condition
        X = data.mean(axis=2).T

        if self.method == "pca":
            model = PCA(n_components=self.n_components)
            model.fit(X)
            explained_variance = (model.explained_variance_ratio_)

        else:
            model = FastICA(n_components=self.n_components,max_iter=1000,tol=1e-3)
            model.fit(X)
            transformed = model.transform(X)
            explained_variance = (np.var(transformed, axis=0)/np.var(transformed, axis=0).sum())

        self.model_ = model
        self.components_ = model.components_.copy()
        self.explained_variance_ = explained_variance

        return self

    def threshold_components(self):

        components = self.components_.copy()

        for i in range(self.n_components):
            threshold = (np.abs(components[i]).mean()+np.abs(components[i]).std())
            mask = (np.abs(components[i]) > threshold)
            components[i] *= mask

        return components

    def transform(self,data: np.ndarray):
        if self.model_ is None:
            raise RuntimeError("Estimator must be fitted first.")

        out_all = {}
        for condition, idx in self.conditions.items():
            X_selected = data[:, :, idx]
            out = self.model_.transform(X_selected.T)

            out_all[condition] = pd.DataFrame(out,columns=[f"Net{i+1}_{condition}"for i in range(self.n_components)])

        return out_all

    def reconstruct_activity(self,data):
        act_all = {}
        for condition, idx in self.conditions.items():
            X_selected = data[:, :, idx]
            out = self.model_.transform(X_selected.T)
            acts = []
            for c in range(self.n_components):
                act = (out[:, c][:, None]@self.components_[c][None, :])
                acts.append(act)

            act_all[condition] = np.stack(acts,axis=2)

        return act_all

    def fit_transform(self,data):
        self.fit(data)

        timecourses = self.transform(data)
        activities = self.reconstruct_activity(data)
        self.result_ = NetworkResult(
            components=self.components_,
            explained_variance=self.explained_variance_,
            timecourses=timecourses,
            reconstructed_activity=activities,
            conditions=self.conditions
        )

        return self.result_
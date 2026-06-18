from dataclasses import dataclass
import pandas as pd
import numpy as np

from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

from src.config import OUT_PATH

@dataclass
class RQAResult:
    metrics: pd.DataFrame
    recurrence_matrices: dict
    distance_matrices: dict
    phase_space_data:dict

    def animate_phase_space(self,condition,output_dir="figures", selected_time_id=None, save=True):
        phase_data = self.phase_space_data[condition]

        trajectory = phase_data["trajectory"]
        time = phase_data["time"]
        pc1 = phase_data["pc1"]
        pc2 = phase_data["pc2"]

        # Optional cropping
        if selected_time_id is not None:
            start, stop = selected_time_id

            trajectory = trajectory[start:stop]
            time = time[start:stop]
            pc1 = pc1[start:stop]
            pc2 = pc2[start:stop]

        time_serie1 = np.column_stack([pc1, time])
        time_serie2 = np.column_stack([pc2, time])

        def update(frame):
            dotemb.set_data([time_serie1[frame,0]], [time_serie2[frame, 1]])
            dottime1.set_data([time_serie1[frame,1]], [time_serie1[frame, 0]])
            dottime2.set_data([time_serie2[frame,1]], [time_serie2[frame, 0]])
            return dotemb,dottime1,dottime2,

        fig = plt.figure(figsize=(11,5))
        axemb = fig.add_axes([0.05, 0.1, 0.45, 0.8]) 
        axemb.set_xlabel('Network 1')
        axemb.set_ylabel('Network 2')
        axemb.set_title('Phase embedding')

        ax2d = fig.add_axes([0.57, 0.1, 0.3, 0.8])    
        ax2d.set_title('Time serie')   
        ax2d.set_xlabel('time')  

        axemb.scatter(time_serie1[:,0], time_serie2[:,1], color='lightgray', alpha=0.5)
        dotemb,  = axemb.plot(time_serie1[0, 0], time_serie2[0, 1], 'ro', markersize=10)

        ax2d.plot(time_serie1[:, 1], time_serie1[:, 0])
        ax2d.plot(time_serie2[:, 1], time_serie2[:, 0])
        ax2d.legend(['Network 1', 'Network 2'])

        dottime1, = ax2d.plot(time_serie1[0, 1], time_serie1[0, 0], 'bo', markersize=6)
        dottime2, = ax2d.plot(time_serie2[0, 1], time_serie2[0, 0], color = 'orange', marker = 'o', markersize=6)

        n_timepoints = time_serie1.shape[0]

        ani = FuncAnimation(fig, update, frames=n_timepoints, interval=100, blit=False)

        if save:
            ani.save(OUT_PATH + f'/phase_{condition}.gif', writer="pillow", fps=15)
            
            plt.close()  

    def plot_recurrence(self,condition,time=None,cmap="Blues", store=True):
        D = self.distance_matrices[condition]
        titles = ["Distance Matrix PC1","Distance Matrix PC2","Distance Matrix Multivariate"]
        fig, ax = plt.subplots(1,3,figsize=(15, 5))
        for i, key in enumerate(["PC1", "PC2", "Multivariate"]):
            im = ax[i].imshow(D[key],aspect="auto",cmap=cmap)
            ax[i].set_title(f"{titles[i]} — {condition}")

            if time is not None:
                ticks = np.arange(0,len(time),max(1, len(time)//10))
                labels = np.round(np.asarray(time)[ticks],2)
                ax[i].set_xticks(ticks,labels,rotation=45)
                ax[i].set_yticks(ticks,labels)

            ax[i].set_xlabel("Time")
            ax[i].set_ylabel("Time")

        fig.colorbar(im,ax=ax,label="Distance")
        plt.tight_layout()

        if store :
            return 0 #TODO

        return fig
            

def angular_distance_matrix(X):
    """Compute pairwise angular distance matrix for a multivariate time series X."""
    n_samples = X.shape[0]
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            u, v = X[i], X[j]
            cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            D[i, j] = np.arccos(cos_theta)  # radians
    return D

def recurrence_plot(ts, eps=0.1):
    """Recurrence plot for 1D time series"""
    N = len(ts)
    D = np.abs(ts.reshape(N,1) - ts.reshape(1,N))
                
    return (D < eps).astype(int), D

def recurrence_plot_multivariate(trajectory, eps=0.1, metric = "euclidean"):
    """Recurrence plot for multivariate trajectory"""
    if metric == "angular":
        D = angular_distance_matrix(trajectory)
        return (D < eps).astype(int), D

    D = cdist(trajectory, trajectory, metric=metric) # angular distance?

    return (D < eps).astype(int), D

def rqa_measures(R, lmin=2, vmin=2):
    """Compute basic RQA measures from recurrence plot"""
    N = R.shape[0]
    RR = np.sum(R) / (N*N)

    # diagonal lines (for DET, ENTR)
    det_points, diag_lengths = 0, []
    recur_points = np.sum(R)

    for k in range(-N+1, N):  # diagonals
        diag = np.diag(R, k=k)
        run = 0
        for val in diag:
            if val == 1:
                run += 1
            else:
                if run >= lmin:
                    det_points += run
                    diag_lengths.append(run)
                run = 0
        if run >= lmin:
            det_points += run
            diag_lengths.append(run)

    DET = det_points / recur_points if recur_points > 0 else 0
    ENTR = (np.sum(diag_lengths * np.log(diag_lengths)) / np.sum(diag_lengths)
            if len(diag_lengths) > 0 else 0)

    # vertical lines (for LAM, TT)
    lam_points, vert_lengths = 0, []
    for j in range(N):  # columns
        run = 0
        for i in range(N):
            if R[i,j] == 1:
                run += 1
            else:
                if run >= vmin:
                    lam_points += run
                    vert_lengths.append(run)
                run = 0
        if run >= vmin:
            lam_points += run
            vert_lengths.append(run)

    LAM = lam_points / recur_points if recur_points > 0 else 0
    TT = np.mean(vert_lengths) if len(vert_lengths) > 0 else 0

    return {"RR": RR, "DET": DET, "ENTR": ENTR, "LAM": LAM, "TT": TT}

def compute_rqa(out_all, conditions, time, nb_compo, method='pca',  summary=True, save=True, eps=0.1, selected_time_id = None): 
    df_rqa = pd.DataFrame()

    recurrence_matrices={}
    distance_matrices = {}
    phase_space_data = {}

    if summary :
        print('--------------------------------------------')
        print(f'Phase Space RQA on {method} with {nb_compo} PCs')
        print('--------------------------------------------')

    for condi in list(conditions.keys()) : 
        if summary :
            print('Condition : ', condi)

        if selected_time_id is not None :
            out1 = out_all[condi].loc[:, f'Net1_{condi}'].values[selected_time_id[0]:selected_time_id[1]]
            out2 = out_all[condi].loc[:, f'Net2_{condi}'].values[selected_time_id[0]:selected_time_id[1]]
            time_array = np.array(time).astype(float)[selected_time_id[0]:selected_time_id[1]]


        else :
            out1 = out_all[condi].loc[:, f'Net1_{condi}'].values
            out2 = out_all[condi].loc[:, f'Net2_{condi}'].values
            time_array = np.array(time).astype(float)
            
        time_serie1 = np.concatenate([out1.reshape(out1.shape[0], 1), time_array.reshape(time_array.shape[0], 1)], axis = 1).astype(float)
        time_serie2 = np.concatenate([out2.reshape(out2.shape[0], 1), time_array.reshape(time_array.shape[0], 1)], axis = 1).astype(float)

        # PC separately
        R1, D1 = recurrence_plot(out1, eps=eps)
        R2, D2 = recurrence_plot(out2, eps=eps)
        
        res1 = rqa_measures(R1)
        res2 = rqa_measures(R2)

        # PC together 
        trajectory = np.vstack([out1, out2]).T 
        R12, D12 = recurrence_plot_multivariate(trajectory, eps=eps)
        res12 = rqa_measures(R12)

        df = pd.DataFrame({f"PC1_{condi}": res1,f"PC2_{condi}": res2,f"Together_{condi}": res12})
        df_rqa = pd.concat([df_rqa, df], axis=1)
        
        recurrence_matrices[condi] = {"PC1": R1,"PC2": R2,"Multivariate": R12}
        distance_matrices[condi] = {"PC1": D1, "PC2": D2,"Multivariate": D12}
        phase_space_data[condi] = {"trajectory": trajectory,"time": time_array,"pc1": out1,"pc2": out2}
    
    return RQAResult(metrics=df_rqa,recurrence_matrices=recurrence_matrices,distance_matrices=distance_matrices, phase_space_data=phase_space_data)


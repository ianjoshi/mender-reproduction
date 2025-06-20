import numpy as np
import scanpy as sc
import squidpy as sq
import time
import anndata as ad
from scipy.spatial.distance import squareform, pdist
import psutil
from sklearn.metrics import *
from sklearn.decomposition import NMF, FastICA, FactorAnalysis
from smender.utils import *
import os
import matplotlib.pyplot as plt
from multiprocessing import Process, Pool
import pandas as pd

class SMENDER_single(object):
    def __init__(self, adata, ct_obs='ct', verbose=0, random_seed=666, ann=None, dim_reduction='pca'):
        """Initialize SMENDER_single with AnnData object and parameters."""
        if adata and isinstance(adata, ad.AnnData) and 'spatial' in adata.obsm:
            self.adata = adata.copy()
            self.set_ct_obs(ct_obs)
            self.nn_mode = None
            self.nn_para = None
            self.count_rep = None
            self.include_self = None
            self.n_scales = None
            self.verbose = verbose
            self.random_seed = random_seed
            self.ann = ann
            self.dim_reduction = dim_reduction.lower()
            if self.dim_reduction not in ['pca', 'nmf', 'ica', 'fa']:
                raise ValueError("dim_reduction must be 'pca', 'nmf', 'ica', or 'fa'")
            if self.ann is not None and not hasattr(self.ann, "build_graph"):
                raise TypeError("Provided ANN object must implement a 'build_graph(adata, k, include_self)' method.")
            self.is_adata_MENDER_preprocess = False
        else:
            print('Please input an AnnData object with spatial coordinates')
            exit(1)
            
    def dump(self, pre=''):
        """Save SMENDER results to h5ad files."""
        if hasattr(self, 'adata_MENDER') and 'MENDER' in self.adata_MENDER.obs:
            if not np.all(self.adata_MENDER.obs.index == self.adata.obs.index):
                print("Warning: Index mismatch between adata_MENDER and adata. Reindexing.")
                self.adata_MENDER = self.adata_MENDER.reindex(self.adata.obs.index)
            self.adata.obs['SMENDER'] = self.adata_MENDER.obs['MENDER'].copy()
            if self.adata.obs['SMENDER'].isna().any():
                print(f"Warning: {self.adata.obs['SMENDER'].isna().sum()} NaN values in SMENDER during dump. Replacing with 'unknown'.")
                self.adata.obs['SMENDER'] = self.adata.obs['SMENDER'].fillna('unknown')
                self.adata.obs['SMENDER'] = self.adata.obs['SMENDER'].astype('category')
        else:
            print("Warning: adata_MENDER or 'MENDER' column not found. Skipping SMENDER assignment.")
        self.adata.write_h5ad(f'{pre}_GEX.h5ad')
        if hasattr(self, 'adata_MENDER'):
            self.adata_MENDER.write_h5ad(f'{pre}_SMENDER.h5ad')
        else:
            print("Warning: adata_MENDER not found. Skipping SMENDER h5ad save.")
        
    def set_ct_obs(self, new_ct):
        """Set cell type observation column."""
        if new_ct not in self.adata.obs:
            print(f"Error: '{new_ct}' not found in adata.obs")
            return
        self.ct_obs = new_ct
        self.ct_unique = np.array(self.adata.obs[self.ct_obs].cat.categories)
        self.ct_array = np.array(self.adata.obs[self.ct_obs])
        if pd.isna(self.ct_array).any():
            print(f"Warning: {pd.isna(self.ct_array).sum()} NaN values in ct_array. Replacing with 'unknown'.")
            self.ct_array = np.where(pd.isna(self.ct_array), 'unknown', self.ct_array)
        self.ct_unique = np.unique(self.ct_array[~pd.isna(self.ct_array)])
    
    def set_smender_para(self, nn_mode='k', nn_para=1, count_rep='s', include_self=True, n_scales=15):
        """Set parameters for neighborhood representation."""
        if nn_mode not in ['ring', 'radius', 'k']:
            print('nn_mode: error in [ring, radius, k]')
            return
        if count_rep not in ['s', 'a']:
            print('count_rep: error in [s, a]')
            return 
        self.nn_mode = nn_mode
        self.nn_para = nn_para
        self.count_rep = count_rep
        self.include_self = include_self
        self.n_scales = n_scales
        
    def run_representation(self, group_norm=False, track_dim_time=False, track_dim_memory=False, track_nn_time=False, track_nn_memory=False):
        """Run multi-scale context representation based on nn_mode."""
        if self.nn_mode == 'ring':
            return self.ring_representation(group_norm, track_dim_time, track_dim_memory, track_nn_time, track_nn_memory)
        elif self.nn_mode == 'radius':
            return self.radius_representation(group_norm, track_dim_time, track_dim_memory, track_nn_time, track_nn_memory)
        elif self.nn_mode == 'k':
            return self.k_representation(group_norm, track_dim_time, track_dim_memory, track_nn_time, track_nn_memory)
        else:
            print('please input [ring, radius, k]')
            return None, 0.0, 0.0, None, 0.0
    
    def print_settings(self):
        """Print current settings."""
        print(f'adata: {np.array(self.adata.shape)}')
        print(f'ct_obs: {self.ct_obs}')
        print(f'nn_mode: {self.nn_mode}')
        print(f'nn_para: {self.nn_para}')
        print(f'count_rep: {self.count_rep}')
        print(f'include_self: {self.include_self}')
        print(f'n_scales: {self.n_scales}')
        
    def estimate_radius(self):
        """Estimate radius for neighborhood representation."""
        spatialmat = self.adata.obsm['spatial']
        if np.isnan(spatialmat).any():
            print(f"Warning: {np.isnan(spatialmat).sum()} NaN values in spatial coordinates. Replacing with 0.")
            spatialmat = np.nan_to_num(spatialmat, nan=0.0)
        min_dist_list = []
        cur_distmat = np.array(squareform(pdist(spatialmat)))
        np.fill_diagonal(cur_distmat, np.inf)
        cur_min_dist = np.min(cur_distmat, axis=0)
        min_dist_list.append(cur_min_dist)
        min_dist_array = np.hstack(min_dist_list)
        neighbor_sz = np.median(min_dist_array)
        if self.verbose:
            print(f'estimated radius: {neighbor_sz}')
        self.estimated_radius = neighbor_sz
        
    def mp_helper(self, cur_scale, track_nn_time=False, track_nn_memory=False):
        """Helper function for parallel representation computation."""
        adata_tmp = self.adata.copy()
        cls_array = self.ct_array
        ME_var_names_np_unique = self.ct_unique
        nn_time = 0.0
        nn_memory = 0.0
        if self.verbose == 1:
            print(f'scale {cur_scale}')
        if track_nn_time:
            nn_start_time = time.time()
        if track_nn_memory:
            nn_start_memory = psutil.Process().memory_info().rss / (1024 ** 2)
        if self.ann:
            self.ann.build_graph(adata_tmp, k=self.nn_para + cur_scale, include_self=self.include_self)
        else:
            sq.gr.spatial_neighbors(adata_tmp, coord_type='generic', n_neighs=self.nn_para + cur_scale, set_diag=self.include_self)
        if track_nn_time:
            nn_time = time.time() - nn_start_time
        if track_nn_memory:
            current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
            nn_memory = max(current_memory - nn_start_memory, 0)
        
        I = adata_tmp.obsp['spatial_connectivities']
        ME_X = np.zeros(shape=(cls_array.shape[0], ME_var_names_np_unique.shape[0]))
        time_start = time.time()
        for i in range(I.shape[0]):
            if self.verbose == 2:
                if i % 1000 == 0:
                    time_end = time.time()
                    print('{0} MEs, time cost {1} s, {2} MEs, {3}s left'.format(
                        i, time_end - time_start, I.shape[0] - i, (I.shape[0] - i) * (time_end - time_start) / 1000))
                    time_start = time.time()
            cur_neighbors = I[i, :].nonzero()[1]
            if len(cur_neighbors) == 0 and not self.include_self:
                print(f"Warning: Cell {i} at scale {cur_scale} has no neighbors. Skipping.")
                continue
            cur_neighbors_cls = cls_array[cur_neighbors]
            cur_cls_unique, cur_cls_count = np.unique(cur_neighbors_cls, return_counts=1)
            cur_cls_idx = [np.where(ME_var_names_np_unique == c)[0][0] for c in cur_cls_unique]
            ME_X[i, cur_cls_idx] = cur_cls_count
        cur_ME_key = f'scale{cur_scale}'
        cur_X = ME_X
        if np.isnan(cur_X).any() or np.isinf(cur_X).any():
            print(f"Warning: {np.isnan(cur_X).sum()} NaN and {np.isinf(cur_X).sum()} inf values in {cur_ME_key}. Replacing with 0.")
            cur_X = np.nan_to_num(cur_X, nan=0.0, posinf=0.0, neginf=0.0)
        return cur_ME_key, cur_X, nn_time, nn_memory
        
    def k_representation_mp(self, mp=200, track_dim_time=False, track_dim_memory=False, track_nn_time=False, track_nn_memory=False):
        """Compute k-nearest neighbor representation in parallel."""
        pool = Pool(mp)
        i_list = np.arange(self.n_scales)
        results = pool.starmap(self.mp_helper, [(i, track_nn_time, track_nn_memory) for i in i_list])
        pool.close()
        pool.join()
        self.ME_key_X_list = [(r[0], r[1]) for r in results]
        nn_time = sum(r[2] for r in results)
        nn_memory = sum(r[3] for r in results)
        for i in range(len(self.ME_key_X_list)):
            self.adata.obsm[self.ME_key_X_list[i][0]] = self.ME_key_X_list[i][1]
        self.generate_ct_representation()
        dim_time, dim_memory = self.preprocess_adata_MENDER(track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        return dim_time, dim_memory, nn_time, nn_memory
        
    def ring_representation(self, group_norm=False, track_dim_time=False, track_dim_memory=False, track_nn_time=False, track_nn_memory=False):
        """Compute ring-based multi-scale representation."""
        cls_array = self.ct_array
        ME_var_names_np_unique = self.ct_unique
        ME_X_prev = np.zeros(shape=(cls_array.shape[0], ME_var_names_np_unique.shape[0]))
        nn_time = 0.0
        nn_memory = 0.0
        for cur_scale in range(self.n_scales):
            if self.verbose == 1:
                print(f'scale {cur_scale}')
            if track_nn_time:
                nn_start_time = time.time()
            if track_nn_memory:
                nn_start_memory = psutil.Process().memory_info().rss / (1024 ** 2)
            if self.ann:
                self.ann.build_graph(self.adata, k=self.nn_para + cur_scale, include_self=self.include_self)
            else:
                sq.gr.spatial_neighbors(self.adata, coord_type='grid', n_neighs=self.nn_para, n_rings=cur_scale + 1, set_diag=self.include_self)
            if track_nn_time:
                nn_time += time.time() - nn_start_time
            if track_nn_memory:
                current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
                nn_memory += max(current_memory - nn_start_memory, 0)
            I = self.adata.obsp['spatial_connectivities']
            ME_X = np.zeros(shape=(cls_array.shape[0], ME_var_names_np_unique.shape[0]))
            time_start = time.time()
            for i in range(I.shape[0]):
                if self.verbose == 2:
                    if i % 1000 == 0:
                        time_end = time.time()
                        print('{0} MEs, time cost {1} s, {2} MEs, {3}s left'.format(
                            i, time_end - time_start, I.shape[0] - i, (I.shape[0] - i) * (time_end - time_start) / 1000))
                        time_start = time.time()
                cur_neighbors = I[i, :].nonzero()[1]
                if len(cur_neighbors) == 0 and not self.include_self:
                    print(f"Warning: Cell {i} at scale {cur_scale} has no neighbors. Skipping.")
                    continue
                cur_neighbors_cls = cls_array[cur_neighbors]
                cur_cls_unique, cur_cls_count = np.unique(cur_neighbors_cls, return_counts=1)
                cur_cls_idx = [np.where(ME_var_names_np_unique == c)[0][0] for c in cur_cls_unique]
                ME_X[i, cur_cls_idx] = cur_cls_count
            cur_ME_key = f'scale{cur_scale}'
            if self.count_rep == 'a':
                cur_X = ME_X
            elif self.count_rep == 's':
                cur_X = ME_X - ME_X_prev
                cur_X = np.maximum(cur_X, 0)
                ME_X_prev = ME_X
            if self.verbose:
                print(f'scale {cur_scale}, median #cells per ring (r={self.nn_para}):', np.median(np.sum(cur_X, axis=1)))
            self.adata.obsm[cur_ME_key] = cur_X.copy()
            if group_norm:
                row_sums = np.sum(self.adata.obsm[cur_ME_key], axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1e-6, row_sums)
                self.adata.obsm[cur_ME_key] = self.adata.obsm[cur_ME_key] / row_sums
                self.adata.obsm[cur_ME_key] = np.where(np.isfinite(self.adata.obsm[cur_ME_key]), self.adata.obsm[cur_ME_key], 0)
            self.adata.obsm[cur_ME_key] = np.where(np.isfinite(self.adata.obsm[cur_ME_key]), self.adata.obsm[cur_ME_key], 0)
            if np.isnan(self.adata.obsm[cur_ME_key]).any() or np.isinf(self.adata.obsm[cur_ME_key]).any():
                print(f"Warning: {np.isnan(self.adata.obsm[cur_ME_key]).sum()} NaN and {np.isinf(self.adata.obsm[cur_ME_key]).sum()} inf values in {cur_ME_key}. Replacing with 0.")
                self.adata.obsm[cur_ME_key] = np.nan_to_num(self.adata.obsm[cur_ME_key], nan=0.0, posinf=0.0, neginf=0.0)
        self.generate_ct_representation()
        dim_time, dim_memory = self.preprocess_adata_MENDER(track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        return None, dim_time, dim_memory, nn_time, nn_memory
            
    def radius_representation(self, group_norm=False, track_dim_time=False, track_dim_memory=False, track_nn_time=False, track_nn_memory=False):
        """Compute radius-based multi-scale representation."""
        cls_array = self.ct_array
        ME_var_names_np_unique = self.ct_unique
        ME_X_prev = np.zeros(shape=(cls_array.shape[0], ME_var_names_np_unique.shape[0]))
        nn_time = 0.0
        nn_memory = 0.0
        for cur_scale in range(self.n_scales):
            if self.verbose == 1:
                print(f'scale {cur_scale}')
            if track_nn_time:
                nn_start_time = time.time()
            if track_nn_memory:
                nn_start_memory = psutil.Process().memory_info().rss / (1024 ** 2)
            if self.ann:
                self.ann.build_graph(self.adata, k=self.nn_para + cur_scale, include_self=self.include_self)
            else:
                sq.gr.spatial_neighbors(self.adata, coord_type='generic', radius=self.nn_para * (cur_scale + 1), set_diag=self.include_self)
            if track_nn_time:
                nn_time += time.time() - nn_start_time
            if track_nn_memory:
                current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
                nn_memory += max(current_memory - nn_start_memory, 0)
            I = self.adata.obsp['spatial_connectivities']
            ME_X = np.zeros(shape=(cls_array.shape[0], ME_var_names_np_unique.shape[0]))
            time_start = time.time()
            for i in range(I.shape[0]):
                if self.verbose == 2:
                    if i % 1000 == 0:
                        time_end = time.time()
                        print('{0} MEs, time cost {1} s, {2} MEs, {3}s left'.format(
                            i, time_end - time_start, I.shape[0] - i, (I.shape[0] - i) * (time_end - time_start) / 1000))
                        time_start = time.time()
                cur_neighbors = I[i, :].nonzero()[1]
                if len(cur_neighbors) == 0 and not self.include_self:
                    print(f"Warning: Cell {i} at scale {cur_scale} has no neighbors. Skipping.")
                    continue
                cur_neighbors_cls = cls_array[cur_neighbors]
                cur_cls_unique, cur_cls_count = np.unique(cur_neighbors_cls, return_counts=1)
                cur_cls_idx = [np.where(ME_var_names_np_unique == c)[0][0] for c in cur_cls_unique]
                ME_X[i, cur_cls_idx] = cur_cls_count
            cur_ME_key = f'scale{cur_scale}'
            if self.count_rep == 'a':
                cur_X = ME_X
            elif self.count_rep == 's':
                cur_X = ME_X - ME_X_prev
                cur_X = np.maximum(cur_X, 0)
                ME_X_prev = ME_X
            if self.verbose:
                print(f'scale {cur_scale}, median #cells per radius (r={self.nn_para * (cur_scale + 1)}):', np.median(np.sum(cur_X, axis=1)))
            self.adata.obsm[cur_ME_key] = cur_X.copy()
            if group_norm:
                row_sums = np.sum(self.adata.obsm[cur_ME_key], axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1e-6, row_sums)
                self.adata.obsm[cur_ME_key] = self.adata.obsm[cur_ME_key] / row_sums
                self.adata.obsm[cur_ME_key] = np.where(np.isfinite(self.adata.obsm[cur_ME_key]), self.adata.obsm[cur_ME_key], 0)
            self.adata.obsm[cur_ME_key] = np.where(np.isfinite(self.adata.obsm[cur_ME_key]), self.adata.obsm[cur_ME_key], 0)
            if np.isnan(self.adata.obsm[cur_ME_key]).any() or np.isinf(self.adata.obsm[cur_ME_key]).any():
                print(f"Warning: {np.isnan(self.adata.obsm[cur_ME_key]).sum()} NaN and {np.isinf(self.adata.obsm[cur_ME_key]).sum()} inf values in {cur_ME_key}. Replacing with 0.")
                self.adata.obsm[cur_ME_key] = np.nan_to_num(self.adata.obsm[cur_ME_key], nan=0.0, posinf=0.0, neginf=0.0)
        self.generate_ct_representation()
        dim_time, dim_memory = self.preprocess_adata_MENDER(track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        return None, dim_time, dim_memory, nn_time, nn_memory
        
    def k_representation(self, group_norm=False, track_dim_time=False, track_dim_memory=False, track_nn_time=False, track_nn_memory=False):
        """Compute k-nearest neighbor multi-scale representation."""
        cls_array = self.ct_array
        ME_var_names_np_unique = self.ct_unique
        ME_X_prev = np.zeros(shape=(cls_array.shape[0], ME_var_names_np_unique.shape[0]))
        nn_time = 0.0
        nn_memory = 0.0
        for cur_scale in range(self.n_scales):
            if self.verbose == 1:
                print(f'scale {cur_scale}')
            if track_nn_time:
                nn_start_time = time.time()
            if track_nn_memory:
                nn_start_memory = psutil.Process().memory_info().rss / (1024 ** 2)
            if self.ann:
                self.ann.build_graph(self.adata, k=self.nn_para + cur_scale, include_self=self.include_self)
            else:
                sq.gr.spatial_neighbors(self.adata, coord_type='generic', n_neighs=self.nn_para + cur_scale, set_diag=self.include_self)
            if track_nn_time:
                nn_time += time.time() - nn_start_time
            if track_nn_memory:
                current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
                nn_memory += max(current_memory - nn_start_memory, 0)
            I = self.adata.obsp['spatial_connectivities']
            ME_X = np.zeros(shape=(cls_array.shape[0], ME_var_names_np_unique.shape[0]))
            time_start = time.time()
            for i in range(I.shape[0]):
                if self.verbose == 2:
                    if i % 1000 == 0:
                        time_end = time.time()
                        print('{0} MEs, time cost {1} s, {2} MEs, {3}s left'.format(
                            i, time_end - time_start, I.shape[0] - i, (I.shape[0] - i) * (time_end - time_start) / 1000))
                        time_start = time.time()
                cur_neighbors = I[i, :].nonzero()[1]
                if len(cur_neighbors) == 0 and not self.include_self:
                    print(f"Warning: Cell {i} at scale {cur_scale} has no neighbors. Skipping.")
                    continue
                cur_neighbors_cls = cls_array[cur_neighbors]
                cur_cls_unique, cur_cls_count = np.unique(cur_neighbors_cls, return_counts=1)
                cur_cls_idx = [np.where(ME_var_names_np_unique == c)[0][0] for c in cur_cls_unique]
                ME_X[i, cur_cls_idx] = cur_cls_count
            cur_ME_key = f'scale{cur_scale}'
            if self.count_rep == 'a':
                cur_X = ME_X
            elif self.count_rep == 's':
                cur_X = ME_X - ME_X_prev
                cur_X = np.maximum(cur_X, 0)
                ME_X_prev = ME_X
            if self.verbose:
                print(f'scale {cur_scale}, median #cells per k (k={self.nn_para + cur_scale}):', np.median(np.sum(cur_X, axis=1)))
            self.adata.obsm[cur_ME_key] = cur_X.copy()
            if group_norm:
                row_sums = np.sum(self.adata.obsm[cur_ME_key], axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1e-6, row_sums)
                self.adata.obsm[cur_ME_key] = self.adata.obsm[cur_ME_key] / row_sums
                self.adata.obsm[cur_ME_key] = np.where(np.isfinite(self.adata.obsm[cur_ME_key]), self.adata.obsm[cur_ME_key], 0)
            self.adata.obsm[cur_ME_key] = np.where(np.isfinite(self.adata.obsm[cur_ME_key]), self.adata.obsm[cur_ME_key], 0)
            if np.isnan(self.adata.obsm[cur_ME_key]).any() or np.isinf(self.adata.obsm[cur_ME_key]).any():
                print(f"Warning: {np.isnan(self.adata.obsm[cur_ME_key]).sum()} NaN and {np.isinf(self.adata.obsm[cur_ME_key]).sum()} inf values in {cur_ME_key}. Replacing with 0.")
                self.adata.obsm[cur_ME_key] = np.nan_to_num(self.adata.obsm[cur_ME_key], nan=0.0, posinf=0.0, neginf=0.0)
        self.generate_ct_representation()
        dim_time, dim_memory = self.preprocess_adata_MENDER(track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        return None, dim_time, dim_memory, nn_time, nn_memory
        
    def generate_ct_representation(self):
        """Generate cell type representation from multi-scale features."""
        ME_var_names_np_unique = self.ct_unique
        whole_feature_list = []
        whole_feature_X = []
        for ct_idx in range(len(ME_var_names_np_unique)):
            rep_list = []
            for i in range(self.n_scales):
                rep_list.append(self.adata.obsm[f'scale{i}'][:, ct_idx])
                whole_feature_list.append(f'ct{ct_idx}scale{i}')
                whole_feature_X.append(self.adata.obsm[f'scale{i}'][:, ct_idx])
            cur_ct_rep = np.array(rep_list).transpose()
            cur_obsm = f'ct{ct_idx}'
            self.adata.obsm[cur_obsm] = cur_ct_rep
        self.adata.obsm['whole'] = np.array(whole_feature_X).transpose()
        adata_feature = ad.AnnData(X=np.array(whole_feature_X).transpose())
        adata_feature.obs_names = self.adata.obs_names
        adata_feature.var_names = whole_feature_list
        adata_feature.obsm['spatial'] = self.adata.obsm['spatial']
        for k in self.adata.obs.keys():
            adata_feature.obs[k] = self.adata.obs[k]
        if 'spatial' in self.adata.uns:
            adata_feature.uns['spatial'] = self.adata.uns['spatial']
        self.adata_MENDER = adata_feature
        if np.isnan(self.adata_MENDER.X).any() or np.isinf(self.adata_MENDER.X).any():
            print(f"Warning: {np.isnan(self.adata_MENDER.X).sum()} NaN and {np.isinf(self.adata_MENDER.X).sum()} inf values in adata_MENDER.X. Replacing with 0.")
            self.adata_MENDER.X = np.nan_to_num(self.adata_MENDER.X, nan=0.0, posinf=0.0, neginf=0.0)
        
    def preprocess_adata_MENDER(self, mode=3, neighbor=True, track_dim_time=False, track_dim_memory=False):
        """Preprocess adata_MENDER for clustering."""
        dim_time = 0.0
        dim_memory = 0.0
        if not hasattr(self, 'is_adata_MENDER_preprocess') or not self.is_adata_MENDER_preprocess:
            if not hasattr(self, 'adata_MENDER'):
                print("Error: adata_MENDER not initialized. Run representation first.")
                return 0.0, 0.0
            if np.any(np.isinf(self.adata_MENDER.X)) or np.any(np.isnan(self.adata_MENDER.X)):
                print(f"Warning: adata_MENDER.X contains {np.sum(np.isinf(self.adata_MENDER.X))} inf and {np.sum(np.isnan(self.adata_MENDER.X))} NaN before preprocessing. Replacing with 0.")
                self.adata_MENDER.X = np.where(np.isfinite(self.adata_MENDER.X), self.adata_MENDER.X, 0)
            max_float = np.finfo(np.float64).max / 100
            min_float = -max_float
            if np.any(self.adata_MENDER.X > max_float) or np.any(self.adata_MENDER.X < min_float):
                print(f"Warning: adata_MENDER.X contains values outside [{min_float}, {max_float}]. Clipping.")
                self.adata_MENDER.X = np.clip(self.adata_MENDER.X, min_float, max_float)
            if mode == 0:
                sc.pp.normalize_total(self.adata_MENDER)
                sc.pp.log1p(self.adata_MENDER)
            elif mode == 1:
                sc.pp.normalize_total(self.adata_MENDER)
            elif mode == 2:
                sc.pp.log1p(self.adata_MENDER)
            elif mode == 3:
                pass
            if np.any(np.isinf(self.adata_MENDER.X)) or np.any(np.isnan(self.adata_MENDER.X)):
                print(f"Warning: adata_MENDER.X contains {np.sum(np.isinf(self.adata_MENDER.X))} inf and {np.sum(np.isnan(self.adata_MENDER.X))} NaN after normalization. Replacing with 0.")
                self.adata_MENDER.X = np.where(np.isfinite(self.adata_MENDER.X), self.adata_MENDER.X, 0)
            if track_dim_time:
                dim_start_time = time.time()
            if track_dim_memory:
                dim_start_memory = psutil.Process().memory_info().rss / (1024 ** 2)
            if self.dim_reduction == 'pca':
                sc.pp.pca(self.adata_MENDER, n_comps=50, random_state=self.random_seed)
                self.adata_MENDER.obsm['X_dim_reduction'] = self.adata_MENDER.obsm['X_pca'].copy()
            elif self.dim_reduction == 'nmf':
                if np.any(self.adata_MENDER.X < 0):
                    print("Warning: NMF requires non-negative input. Clipping negative values to 0.")
                    self.adata_MENDER.X = np.clip(self.adata_MENDER.X, 0, np.inf)
                model = NMF(n_components=50, init='random', random_state=self.random_seed, max_iter=1000)
                W = model.fit_transform(self.adata_MENDER.X)
                self.adata_MENDER.obsm['X_dim_reduction'] = W
            elif self.dim_reduction == 'ica':
                model = FastICA(n_components=50, random_state=self.random_seed, max_iter=1000)
                X_ica = model.fit_transform(self.adata_MENDER.X)
                self.adata_MENDER.obsm['X_dim_reduction'] = X_ica
            elif self.dim_reduction == 'fa':
                model = FactorAnalysis(n_components=50, random_state=self.random_seed)
                X_fa = model.fit_transform(self.adata_MENDER.X)
                self.adata_MENDER.obsm['X_dim_reduction'] = X_fa
            else:
                raise ValueError("dim_reduction must be 'pca', 'nmf', 'ica', or 'fa'")
            if np.isnan(self.adata_MENDER.obsm['X_dim_reduction']).any() or np.isinf(self.adata_MENDER.obsm['X_dim_reduction']).any():
                print(f"Warning: {np.isnan(self.adata_MENDER.obsm['X_dim_reduction']).sum()} NaN and {np.isinf(self.adata_MENDER.obsm['X_dim_re Узнать большеreduction']).sum()} inf values in X_dim_reduction. Replacing with 0.")
                self.adata_MENDER.obsm['X_dim_reduction'] = np.nan_to_num(self.adata_MENDER.obsm['X_dim_reduction'], nan=0.0, posinf=0.0, neginf=0.0)
            if track_dim_time:
                dim_time = time.time() - dim_start_time
            if track_dim_memory:
                current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
                dim_memory = max(current_memory - dim_start_memory, 0)
            if neighbor:
                try:
                    sc.pp.neighbors(self.adata_MENDER, use_rep='X_dim_reduction', n_neighbors=30, random_state=self.random_seed)
                    disconnected = np.sum(self.adata_MENDER.obsp['connectivities'].sum(axis=1) == 0)
                    if disconnected > 0:
                        print(f"Warning: {disconnected} cells are disconnected in the neighbor graph. Increasing n_neighbors.")
                        sc.pp.neighbors(self.adata_MENDER, use_rep='X_dim_reduction', n_neighbors=50, random_state=self.random_seed)
                except Exception as e:
                    print(f"Error in neighbor graph construction: {e}. Falling back to default k=15.")
                    sc.pp.neighbors(self.adata_MENDER, use_rep='X_dim_reduction', n_neighbors=15, random_state=self.random_seed)
                if 'connectivities' not in self.adata_MENDER.obsp:
                    print("Error: Neighbor graph construction failed.")
                    return 0.0, 0.0
            self.is_adata_MENDER_preprocess = True
        return dim_time, dim_memory
    
    def run_clustering_normal(self, target_k, run_umap=False, if_reprocess=True, track_dim_time=False, track_dim_memory=False):
        """Run Leiden clustering with resolution search, handling NaN."""
        dim_time = 0.0
        dim_memory = 0.0
        if not hasattr(self, 'adata_MENDER'):
            print("Error: adata_MENDER not initialized. Run representation first.")
            return 0.0, 0.0
        if if_reprocess:
            dim_time, dim_memory = self.preprocess_adata_MENDER(mode=3, neighbor=True, track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        if run_umap:
            try:
                sc.tl.umap(self.adata_MENDER, obsm_key='X_dim_reduction')
                self.adata_MENDER.obsm['X_MENDERMAP2D'] = self.adata_MENDER.obsm['X_umap'].copy()
            except Exception as e:
                print(f"UMAP failed: {e}. Skipping UMAP.")
        if target_k > 0:
            try:
                res = res_search(self.adata_MENDER, target_k=target_k, random_state=self.random_seed, use_rep='X_dim_reduction')
                sc.tl.leiden(self.adata_MENDER, resolution=res, key_added=f'MENDER_leiden_k{target_k}', random_state=self.random_seed)
                self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_leiden_k{target_k}'].copy()
            except Exception as e:
                print(f"Leiden clustering failed with target_k={target_k}: {e}. Falling back to resolution=1.0.")
                sc.tl.leiden(self.adata_MENDER, resolution=1.0, key_added=f'MENDER_leiden_fallback', random_state=self.random_seed)
                self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_leiden_fallback'].copy()
        elif target_k < 0:
            res = -target_k
            try:
                sc.tl.leiden(self.adata_MENDER, resolution=res, key_added=f'MENDER_leiden_res{res}', random_state=self.random_seed)
                self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_leiden_res{res}'].copy()
            except Exception as e:
                print(f"Leiden clustering failed with resolution={res}: {e}. Falling back to resolution=1.0.")
                sc.tl.leiden(self.adata_MENDER, resolution=1.0, key_added=f'MENDER_leiden_fallback', random_state=self.random_seed)
                self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_leiden_fallback'].copy()
        else:
            print('please input a valid target_k')
            return 0.0, 0.0
        if 'MENDER' not in self.adata_MENDER.obs:
            print("Error: Leiden clustering failed to produce 'MENDER' column.")
            self.adata_MENDER.obs['MENDER'] = pd.Series(['unknown'] * len(self.adata_MENDER), index=self.adata_MENDER.obs.index)
        elif self.adata_MENDER.obs['MENDER'].isna().any():
            print(f"Warning: {self.adata_MENDER.obs['MENDER'].isna().sum()} NaN values in MENDER. Replacing with 'unknown'.")
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs['MENDER'].fillna('unknown')
        self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs['MENDER'].astype('category')
        return dim_time, dim_memory
    
    def run_clustering_mclust(self, target_k, run_umap=False, track_dim_time=False, track_dim_memory=False):
        """Run mclust clustering."""
        dim_time = 0.0
        dim_memory = 0.0
        if not hasattr(self, 'adata_MENDER'):
            print("Error: adata_MENDER not initialized. Run representation first.")
            return 0.0, 0.0
        if run_umap:
            dim_time, dim_memory = self.preprocess_adata_MENDER(mode=3, neighbor=True, track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
            try:
                sc.tl.umap(self.adata_MENDER, obsm_key='X_dim_reduction')
                self.adata_MENDER.obsm['X_MENDERMAP2D'] = self.adata_MENDER.obsm['X_umap'].copy()
            except Exception as e:
                print(f"UMAP failed: {e}. Skipping UMAP.")
        else:
            dim_time, dim_memory = self.preprocess_adata_MENDER(mode=3, neighbor=False, track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        if target_k > 0:
            try:
                self.adata_MENDER = STAGATE.mclust_R(self.adata_MENDER, used_obsm='X_dim_reduction', num_cluster=target_k)
                self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_mclust_k{target_k}'].copy()
            except Exception as e:
                print(f"mclust failed with target_k={target_k}: {e}. Assigning 'unknown'.")
                self.adata_MENDER.obs['MENDER'] = pd.Series(['unknown'] * len(self.adata_MENDER), index=self.adata_MENDER.obs.index)
            if self.adata_MENDER.obs['MENDER'].isna().any():
                print(f"Warning: {self.adata_MENDER.obs['MENDER'].isna().sum()} NaN values in MENDER after mclust. Replacing with 'unknown'.")
                self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs['MENDER'].fillna('unknown')
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs['MENDER'].astype('category')
        else:
            print('please input a valid target_k')
            return 0.0, 0.0
        return dim_time, dim_memory
    
    def output_cluster(self, dirname, obs):
        """Output clustering results as spatial plots."""
        if not hasattr(self, 'adata_MENDER'):
            print("Error: adata_MENDER not initialized.")
            return
        sc.pl.embedding(self.adata_MENDER, basis='X_MENDERMAP2D', color=obs, save=f'_umap_{obs}')
        path = dirname
        os.makedirs(f'figures/spatial_{path}', exist_ok=True)
        adata_feature = self.adata_MENDER
        for i in range(len(self.batch_list)):
            cur_batch = self.batch_list[i]
            cur_a = adata_feature[adata_feature.obs[self.batch_obs] == cur_batch]
            ax = sc.pl.embedding(cur_a, basis='spatial', color=obs, show=False, title=cur_batch, save=None)
            ax.axis('equal')
            plt.savefig(f'figures/spatial_{path}/{cur_batch}.png', dpi=200, bbox_inches='tight', transparent=True)
            plt.close()

    def output_cluster_single(self, obs, idx=0):
        """Output clustering results for a single batch."""
        if not hasattr(self, 'adata_MENDER'):
            print("Error: adata_MENDER not initialized.")
            return
        cur_batch = self.batch_list[idx]
        adata_feature = self.adata_MENDER
        cur_a = adata_feature[adata_feature.obs[self.batch_obs] == cur_batch]
        sc.pl.embedding(cur_a, basis='spatial', color=obs, show=True, title=cur_batch)
        
    def output_cluster_all(self, obs='MENDER', obs_gt='gt'):
        """Output clustering results with metrics for all batches."""
        if not hasattr(self, 'adata_MENDER'):
            print("Error: adata_MENDER not initialized.")
            return
        sc.pl.embedding(self.adata_MENDER, basis='spatial', color=obs)
        self.adata_MENDER.obs[self.batch_obs] = self.adata_MENDER.obs[self.batch_obs].astype('category')
        for si in self.adata_MENDER.obs[self.batch_obs].cat.categories:
            cur_a = self.adata_MENDER[self.adata_MENDER.obs[self.batch_obs] == si]
            if obs_gt in cur_a.obs:
                nmi = compute_NMI(cur_a, obs_gt, obs)
                ari = compute_ARI(cur_a, obs_gt, obs)
                pas = compute_PAS(cur_a, obs)
                chaos = compute_CHAOS(cur_a, obs)
                nmi = np.round(nmi, 3)
                ari = np.round(ari, 3)
                pas = np.round(pas, 3)
                chaos = np.round(chaos, 3)
                title = f'{si}\n nmi:{nmi} ari:{ari}\n pas:{pas} chaos:{chaos}'
            else:
                pas = compute_PAS(cur_a, obs)
                chaos = compute_CHAOS(cur_a, obs)
                pas = np.round(pas, 3)
                chaos = np.round(chaos, 3)
                title = f'{si}\n pas:{pas} chaos:{chaos}'
            ax = sc.pl.embedding(cur_a, basis='spatial', color=obs, show=False)
            ax.axis('equal')
            ax.set_title(title)
            plt.savefig(f'figures/spatial_{si}.png', dpi=200, bbox_inches='tight', transparent=True)
            plt.close()

class SMENDER(object):
    def __init__(self, adata, batch_obs, ct_obs='ct', verbose=0, random_seed=666, ann=None, dim_reduction='pca'):
        """Initialize SMENDER with AnnData object and parameters."""
        if batch_obs not in adata.obs:
            print(f"Error: '{batch_obs}' not found in adata.obs")
            exit(1)
        self.adata = adata[:, 0:0].copy()
        self.batch_obs = batch_obs
        self.ct_obs = ct_obs
        self.ct_unique = np.array(self.adata.obs[self.ct_obs].cat.categories)
        self.verbose = verbose
        self.random_seed = random_seed
        self.ann_class = ann
        self.dim_reduction = dim_reduction.lower()
        if self.dim_reduction not in ['pca', 'nmf', 'ica', 'fa']:
            raise ValueError("dim_reduction must be 'pca', 'nmf', 'ica', or 'fa'")
        self.nn_mode = None
        self.nn_para = None
        self.count_rep = None
        self.include_self = None
        self.n_scales = None
        self.group_norm = None
        self.adata_list = []
        self.batch_list = np.array(self.adata.obs[self.batch_obs].cat.categories)
        for b in self.batch_list:
            cur_a = self.adata[self.adata.obs[self.batch_obs] == b]
            self.adata_list.append(cur_a)
        if self.ann_class is not None and not hasattr(self.ann_class, "build_graph"):
            raise TypeError("Provided ANN object must implement a 'build_graph(adata, k, include_self)' method.")
        # Initialize tracking attributes
        self.smender_time = 0.0
        self.smender_memory = 0.0
        self.dim_reduction_time = 0.0
        self.dim_reduction_memory = 0.0
        self.nn_time = 0.0
        self.nn_memory = 0.0
        self.is_tracking_smender_time = False
        self.is_tracking_smender_memory = False
        self.is_tracking_dim_reduction_time = False
        self.is_tracking_dim_reduction_memory = False
        self.is_tracking_nn_time = False
        self.is_tracking_nn_memory = False
        self.smender_time_start = None
        self.smender_memory_start = None
        self.dim_reduction_time_start = None
        self.dim_reduction_memory_start = None
        self.nn_time_start = None
        self.nn_memory_start = None

    def start_smender_timing(self):
        """Start timing for the entire SMENDER operation."""
        if self.is_tracking_smender_time:
            print("SMENDER timing already started.")
            return
        self.is_tracking_smender_time = True
        self.smender_time_start = time.time()
        
    def stop_smender_timing(self):
        """Stop timing for the entire SMENDER operation and return total time."""
        if not self.is_tracking_smender_time:
            print("SMENDER timing not started.")
            return self.smender_time
        self.is_tracking_smender_time = False
        elapsed = time.time() - self.smender_time_start
        self.smender_time += elapsed
        self.smender_time_start = None
        return self.smender_time
        
    def start_smender_memory(self):
        """Start memory tracking for the entire SMENDER operation."""
        if self.is_tracking_smender_memory:
            print("SMENDER memory tracking already started.")
            return
        self.is_tracking_smender_memory = True
        self.smender_memory_start = psutil.Process().memory_info().rss / (1024 ** 2)
        
    def stop_smender_memory(self):
        """Stop memory tracking for the entire SMENDER operation and return total usage."""
        if not self.is_tracking_smender_memory:
            print("SMENDER memory tracking not started.")
            return self.smender_memory
        self.is_tracking_smender_memory = False
        current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
        usage = current_memory - self.smender_memory_start
        self.smender_memory += max(usage, 0)
        self.smender_memory_start = None
        return self.smender_memory
        
    def start_dim_reduction_timing(self):
        """Start timing for dimensionality reduction across all batches."""
        if self.is_tracking_dim_reduction_time:
            print("Dimensionality reduction timing already started.")
            return
        self.is_tracking_dim_reduction_time = True
        self.dim_reduction_time_start = time.time()
        
    def stop_dim_reduction_timing(self):
        """Stop timing for dimensionality reduction and return total time."""
        if not self.is_tracking_dim_reduction_time:
            print("Dimensionality reduction timing not started.")
            return self.dim_reduction_time
        self.is_tracking_dim_reduction_time = False
        elapsed = time.time() - self.dim_reduction_time_start
        self.dim_reduction_time += elapsed
        self.dim_reduction_time_start = None
        return self.dim_reduction_time
        
    def start_dim_reduction_memory(self):
        """Start memory tracking for dimensionality reduction across all batches."""
        if self.is_tracking_dim_reduction_memory:
            print("Dimensionality reduction memory tracking already started.")
            return
        self.is_tracking_dim_reduction_memory = True
        self.dim_reduction_memory_start = psutil.Process().memory_info().rss / (1024 ** 2)
        
    def stop_dim_reduction_memory(self):
        """Stop memory tracking for dimensionality reduction and return total usage."""
        if not self.is_tracking_dim_reduction_memory:
            print("Dimensionality reduction memory tracking not started.")
            return self.dim_reduction_memory
        self.is_tracking_dim_reduction_memory = False
        current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
        usage = current_memory - self.dim_reduction_memory_start
        self.dim_reduction_memory += max(usage, 0)
        self.dim_reduction_memory_start = None
        return self.dim_reduction_memory
        
    def start_nn_timing(self):
        """Start timing for NN/non-NN operations across all batches."""
        if self.is_tracking_nn_time:
            print("NN timing already started.")
            return
        self.is_tracking_nn_time = True
        self.nn_time_start = time.time()
        
    def stop_nn_timing(self):
        """Stop timing for NN/non-NN operations and return total time."""
        if not self.is_tracking_nn_time:
            print("NN timing not started.")
            return self.nn_time
        self.is_tracking_nn_time = False
        elapsed = time.time() - self.nn_time_start
        self.nn_time += elapsed
        self.nn_time_start = None
        return self.nn_time
        
    def start_nn_memory(self):
        """Start memory tracking for NN/non-NN operations across all batches."""
        if self.is_tracking_nn_memory:
            print("NN memory tracking already started.")
            return
        self.is_tracking_nn_memory = True
        self.nn_memory_start = psutil.Process().memory_info().rss / (1024 ** 2)
        
    def stop_nn_memory(self):
        """Stop memory tracking for NN/non-NN operations and return total usage."""
        if not self.is_tracking_nn_memory:
            print("NN memory tracking not started.")
            return self.nn_memory
        self.is_tracking_nn_memory = False
        current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
        usage = current_memory - self.nn_memory_start
        self.nn_memory += max(usage, 0)
        self.nn_memory_start = None
        return self.nn_memory
    
    def prepare(self):
        """Prepare data for SMENDER processing."""
        self.adata_list = []
        self.batch_list = np.array(self.adata.obs[self.batch_obs].cat.categories)
        for b in self.batch_list:
            cur_a = self.adata[self.adata.obs[self.batch_obs] == b]
            self.adata_list.append(cur_a)
            
    def set_ct_obs(self, new_ct):
        """Set cell type observation column."""
        if new_ct not in self.adata.obs:
            print('Please input a valid cell type obs')
            return
        else:
            self.ct_obs = new_ct
            self.ct_unique = np.array(self.adata.obs[self.ct_obs].cat.categories)
            if pd.isna(self.adata.obs[self.ct_obs]).any():
                print(f"Warning: {pd.isna(self.adata.obs[self.ct_obs]).sum()} NaN values in {self.ct_obs}. Replacing with 'unknown'.")
                self.adata.obs[self.ct_obs] = self.adata.obs[self.ct_obs].fillna('unknown')
            self.ct_unique = np.unique(self.adata.obs[self.ct_obs])
    
    def set_MENDER_para(self, nn_mode='k', nn_para=1, count_rep='s', include_self=True, n_scales=15):
        """Set parameters for all SMENDER instances."""
        if nn_mode not in ['ring', 'radius', 'k']:
            print('nn_mode: please input in [ring, radius, k]')
            return
        if count_rep not in ['s', 'a']:
            print('count_rep: please input in [s, a]')
            return 
        self.nn_mode = nn_mode
        self.nn_para = nn_para
        self.count_rep = count_rep
        self.include_self = include_self
        self.n_scales = n_scales
        
    def run_representation(self, group_norm=False):
        """Run representation sequentially."""
        print('for faster version, use run_representation_mp')
        adata_MENDER_list = []
        total_dim_time = 0.0
        total_dim_memory = 0.0
        total_nn_time = 0.0
        total_nn_memory = 0.0
        for i in range(len(self.batch_list)): 
            cur_batch_name = self.batch_list[i]
            cur_batch_adata = self.adata_list[i]
            print(f'total batch: {len(self.batch_list)}, running batch {cur_batch_name}')
            cur_MENDER = SMENDER_single(
                cur_batch_adata,
                ct_obs=self.ct_obs,
                verbose=self.verbose,
                random_seed=self.random_seed,
                ann=self.ann_class() if self.ann_class else None,
                dim_reduction=self.dim_reduction)
            cur_MENDER.set_smender_para(
                nn_mode=self.nn_mode,
                nn_para=self.nn_para,
                count_rep=self.count_rep,
                include_self=self.include_self,
                n_scales=self.n_scales
            )
            cur_MENDER.ct_unique = self.ct_unique
            _, dim_time, dim_memory, nn_time, nn_memory = cur_MENDER.run_representation(
                group_norm=group_norm,
                track_dim_time=self.is_tracking_dim_reduction_time,
                track_dim_memory=self.is_tracking_dim_reduction_memory,
                track_nn_time=self.is_tracking_nn_time,
                track_nn_memory=self.is_tracking_nn_memory
            )
            total_dim_time += dim_time
            total_dim_memory += dim_memory
            total_nn_time += nn_time
            total_nn_memory += nn_memory
            cur_adata_MENDER = cur_MENDER.adata_MENDER.copy()
            adata_MENDER_list.append(cur_adata_MENDER)
        self.adata_MENDER_list = adata_MENDER_list
        if self.is_tracking_dim_reduction_time:
            self.dim_reduction_time += total_dim_time
        if self.is_tracking_dim_reduction_memory:
            self.dim_reduction_memory += total_dim_memory
        if self.is_tracking_nn_time:
            self.nn_time += total_nn_time
        if self.is_tracking_nn_memory:
            self.nn_memory += total_nn_memory
    
    @staticmethod
    def mp_helper(args):
        """Helper function for parallel representation."""
        i, batch_name, batch_adata, ct_obs, verbose, random_seed, ann_class, dim_reduction, nn_mode, nn_para, count_rep, include_self, n_scales, group_norm, track_dim_time, track_dim_memory, track_nn_time, track_nn_memory, ct_unique = args
        print(f'total batch: {i+1}, running batch {batch_name}')
        ann_instance = ann_class() if ann_class else None
        cur_MENDER = SMENDER_single(
            batch_adata,
            ct_obs=ct_obs,
            verbose=verbose,
            random_seed=random_seed,
            ann=ann_instance,
            dim_reduction=dim_reduction)
        cur_MENDER.set_smender_para(
            nn_mode=nn_mode,
            nn_para=nn_para,
            count_rep=count_rep,
            include_self=include_self,
            n_scales=n_scales
        )
        cur_MENDER.ct_unique = ct_unique
        _, dim_time, dim_memory, nn_time, nn_memory = cur_MENDER.run_representation(
            group_norm=group_norm,
            track_dim_time=track_dim_time,
            track_dim_memory=track_dim_memory,
            track_nn_time=track_nn_time,
            track_nn_memory=track_nn_memory
        )
        cur_adata_MENDER = cur_MENDER.adata_MENDER.copy()
        return cur_adata_MENDER, dim_time, dim_memory, nn_time, nn_memory
        
    def estimate_radius(self):
        """Estimate radius for each batch."""
        for i in range(len(self.batch_list)):
            cur_batch_name = self.batch_list[i]
            cur_batch_adata = self.adata_list[i]
            cur_MENDER = SMENDER_single(cur_batch_adata, ct_obs=self.ct_obs, verbose=self.verbose, random_seed=self.random_seed)
            cur_MENDER.estimate_radius()
            print(f'{cur_batch_name}: estimated radius: {cur_MENDER.estimated_radius}')
        
    def run_representation_mp(self, mp=200, group_norm=False):
        """Run representation in parallel across batches."""
        print('default number of processes is 200')
        self.group_norm = group_norm
        adata_MENDER_list = []
        total_dim_time = 0.0
        total_dim_memory = 0.0
        total_nn_time = 0.0
        total_nn_memory = 0.0
        i_list = np.arange(len(self.batch_list))
        pool = Pool(mp)
        args = [
            (
                i,
                self.batch_list[i],
                self.adata_list[i],
                self.ct_obs,
                self.verbose,
                self.random_seed,
                self.ann_class,
                self.dim_reduction,
                self.nn_mode,
                self.nn_para,
                self.count_rep,
                self.include_self,
                self.n_scales,
                self.group_norm,
                self.is_tracking_dim_reduction_time,
                self.is_tracking_dim_reduction_memory,
                self.is_tracking_nn_time,
                self.is_tracking_nn_memory,
                self.ct_unique
            ) for i in i_list
        ]
        results = pool.map(self.mp_helper, args)
        pool.close()
        pool.join()
        adata_MENDER_list = [r[0] for r in results]
        total_dim_time = sum(r[1] for r in results)
        total_dim_memory = sum(r[2] for r in results)
        total_nn_time = sum(r[3] for r in results)
        total_nn_memory = sum(r[4] for r in results)
        if len(adata_MENDER_list) > 1:
            try:
                self.adata_MENDER = adata_MENDER_list[0].concatenate(adata_MENDER_list[1:], index_unique=None)
                if not np.all(self.adata_MENDER.obs.index == self.adata.obs.index):
                    print("Warning: Index mismatch after concatenation. Reindexing.")
                    self.adata_MENDER = self.adata_MENDER.reindex(self.adata.obs.index)
            except Exception as e:
                print(f"Concatenation failed: {e}. Using first batch only.")
                self.adata_MENDER = adata_MENDER_list[0].copy()
        else:
            self.adata_MENDER = adata_MENDER_list[0].copy()
        self.adata_MENDER_dump = self.adata_MENDER.copy()
        if np.isnan(self.adata_MENDER.X).any() or np.isinf(self.adata_MENDER.X).any():
            print(f"Warning: {np.isnan(self.adata_MENDER.X).sum()} NaN and {np.isinf(self.adata_MENDER.X).sum()} inf values in concatenated adata_MENDER.X. Replacing with 0.")
            self.adata_MENDER.X = np.nan_to_num(self.adata_MENDER.X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.is_tracking_dim_reduction_time:
            self.dim_reduction_time += total_dim_time
        if self.is_tracking_dim_reduction_memory:
            self.dim_reduction_memory += total_dim_memory
        if self.is_tracking_nn_time:
            self.nn_time += total_nn_time
        if self.is_tracking_nn_memory:
            self.nn_memory += total_nn_memory
    
    def refresh_adata_MENDER(self):
        """Refresh adata_MENDER from dump."""
        if not hasattr(self, 'adata_MENDER_dump'):
            print("Error: adata_MENDER_dump not found.")
            return
        del self.adata_MENDER
        self.adata_MENDER = self.adata_MENDER_dump.copy()
        self.is_adata_MENDER_preprocess = False
        
    def preprocess_adata_MENDER(self, mode=3, neighbor=True, track_dim_time=False, track_dim_memory=False):
        """Preprocess adata_MENDER for clustering."""
        dim_time, dim_memory = SMENDER_single.preprocess_adata_MENDER(
            self, mode=mode, neighbor=neighbor,
            track_dim_time=track_dim_time,
            track_dim_memory=track_dim_memory
        )
        if self.is_tracking_dim_reduction_time:
            self.dim_reduction_time += dim_time
        if self.is_tracking_dim_reduction_memory:
            self.dim_reduction_memory += dim_memory
        return dim_time, dim_memory
        
    def run_clustering_normal(self, target_k, run_umap=False, if_reprocess=True):
        """Run clustering for all SMENDER instances."""
        if not hasattr(self, 'adata_MENDER'):
            print("Error: adata_MENDER not initialized. Run representation first.")
            return 0.0, 0.0
        dim_time, dim_memory = SMENDER_single.run_clustering_normal(
            self, target_k, run_umap=run_umap, if_reprocess=if_reprocess,
            track_dim_time=self.is_tracking_dim_reduction_time,
            track_dim_memory=self.is_tracking_dim_reduction_memory
        )
        if self.is_tracking_dim_reduction_time:
            self.dim_reduction_time += dim_time
        if self.is_tracking_dim_reduction_memory:
            self.dim_reduction_memory += dim_memory
        if 'MENDER' in self.adata_MENDER.obs and self.adata_MENDER.obs['MENDER'].isna().any():
            print(f"Warning: {self.adata_MENDER.obs['MENDER'].isna().sum()} NaN values in concatenated MENDER. Replacing with 'unknown'.")
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs['MENDER'].fillna('unknown')
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs['MENDER'].astype('category')
        return dim_time, dim_memory
        
    def run_clustering_mclust(self, target_k, run_umap=False):
        """Run mclust clustering for all SMENDER instances."""
        if not hasattr(self, 'adata_MENDER'):
            print("Error: adata_MENDER not initialized. Run representation first.")
            return 0.0, 0.0
        dim_time, dim_memory = SMENDER_single.run_clustering_mclust(
            self, target_k, run_umap=run_umap,
            track_dim_time=self.is_tracking_dim_reduction_time,
            track_dim_memory=self.is_tracking_dim_reduction_memory
        )
        if self.is_tracking_dim_reduction_time:
            self.dim_reduction_time += dim_time
        if self.is_tracking_dim_reduction_memory:
            self.dim_reduction_memory += dim_memory
        if 'MENDER' in self.adata_MENDER.obs and self.adata_MENDER.obs['MENDER'].isna().any():
            print(f"Warning: {self.adata_MENDER.obs['MENDER'].isna().sum()} NaN values in concatenated MENDER. Replacing with 'unknown'.")
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs['MENDER'].fillna('unknown')
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs['MENDER'].astype('category')
        return dim_time, dim_memory
        
    def output_cluster(self, dirname, obs):
        """Output clustering results as spatial plots."""
        if not hasattr(self, 'adata_MENDER'):
            print("Error: adata_MENDER not initialized.")
            return
        sc.pl.embedding(self.adata_MENDER, basis='X_MENDERMAP2D', color=obs, save=f'_umap_{obs}')
        path = dirname
        os.makedirs(f'figures/spatial_{path}', exist_ok=True)
        adata_feature = self.adata_MENDER
        for i in range(len(self.batch_list)):
            cur_batch = self.batch_list[i]
            cur_a = adata_feature[adata_feature.obs[self.batch_obs] == cur_batch]
            ax = sc.pl.embedding(cur_a, basis='spatial', color=obs, show=False, title=cur_batch, save=None)
            ax.axis('equal')
            plt.savefig(f'figures/spatial_{path}/{cur_batch}.png', dpi=200, bbox_inches='tight', transparent=True)
            plt.close()
    
    def output_cluster_single(self, obs, idx=0):
        """Output clustering results for a single batch."""
        if not hasattr(self, 'adata_MENDER'):
            print("Error: adata_MENDER not initialized.")
            return
        cur_batch = self.batch_list[idx]
        adata_feature = self.adata_MENDER
        cur_a = adata_feature[adata_feature.obs[self.batch_obs] == cur_batch]
        sc.pl.embedding(cur_a, basis='spatial', color=obs, show=True, title=cur_batch)
        
    def output_cluster_all(self, obs='MENDER', obs_gt='gt'):
        """Output clustering results with metrics for all batches."""
        if not hasattr(self, 'adata_MENDER'):
            print("Error: adata_MENDER not initialized.")
            return
        sc.pl.embedding(self.adata_MENDER, basis='spatial', color=obs)
        self.adata_MENDER.obs[self.batch_obs] = self.adata_MENDER.obs[self.batch_obs].astype('category')
        for si in self.adata_MENDER.obs[self.batch_obs].cat.categories:
            cur_a = self.adata_MENDER[self.adata_MENDER.obs[self.batch_obs] == si]
            if obs_gt in cur_a.obs:
                nmi = compute_NMI(cur_a, obs_gt, obs)
                ari = compute_ARI(cur_a, obs_gt, obs)
                pas = compute_PAS(cur_a, obs)
                chaos = compute_CHAOS(cur_a, obs)
                nmi = np.round(nmi, 3)
                ari = np.round(ari, 3)
                pas = np.round(pas, 3)
                chaos = np.round(chaos, 3)
                title = f'{si}\n nmi:{nmi} ari:{ari}\n pas:{pas} chaos:{chaos}'
            else:
                pas = compute_PAS(cur_a, obs)
                chaos = compute_CHAOS(cur_a, obs)
                pas = np.round(pas, 3)
                chaos = np.round(chaos, 3)
                title = f'{si}\n pas:{pas} chaos:{chaos}'
            ax = sc.pl.embedding(cur_a, basis='spatial', color=obs, show=False)
            ax.axis('equal')
            ax.set_title(title)
            plt.savefig(f'figures/spatial_{si}.png', dpi=200, bbox_inches='tight', transparent=True)
            plt.close()
import numpy as np
import scanpy as sc
import squidpy as sq
from scipy.spatial.distance import *
import time
import anndata as ad
from sklearn.metrics import *
from sklearn.decomposition import NMF, FastICA, FactorAnalysis
from smender.utils import *
import os
import matplotlib.pyplot as plt
import psutil

class SMENDER_single(object):
    def __init__(self, adata, ct_obs='ct', verbose=0, random_seed=666, ann=None, dim_reduction='pca'):
        if adata and isinstance(adata, ad.AnnData) and 'spatial' in adata.obsm:
            self.adata = adata
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
        else:
            print('Please input an anndata object with spatial coordinates')
            exit(1)
            
    def dump(self, pre=''):
        self.adata.obs['MENDER'] = self.adata_MENDER.obs['MENDER'].copy()
        self.adata.write_h5ad(f'{pre}_GEX.h5ad')
        self.adata_MENDER.write_h5ad(f'{pre}_MENDER.h5ad')
        
    def set_ct_obs(self, new_ct):
        if new_ct not in self.adata.obs:
            print('Please input a valid cell type obs')
            return
        else:
            self.ct_obs = new_ct
            self.ct_unique = np.array(self.adata.obs[self.ct_obs].cat.categories)
            self.ct_array = np.array(self.adata.obs[self.ct_obs])
    
    def set_MENDER_para(self, nn_mode='k', nn_para=1, count_rep='s', include_self=True, n_scales=15):
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
        
    def run_representation(self, group_norm=False, track_dim_time=False, track_dim_memory=False, track_nn_time=False, track_nn_memory=False):
        if self.nn_mode == 'ring':
            return self.ring_representation(group_norm, track_dim_time, track_dim_memory, track_nn_time, track_nn_memory)
        elif self.nn_mode == 'radius':
            return self.radius_representation(group_norm, track_dim_time, track_dim_memory, track_nn_time, track_nn_memory)
        elif self.nn_mode == 'k':
            return self.k_representation(track_dim_time, track_dim_memory, track_nn_time, track_nn_memory)
        else:
            print('Please input [ring, radius, k]')
            return None, 0.0, 0.0, 0.0, 0.0
    
    def print_settings(self):
        print(f'adata: {self.adata.shape}')
        print(f'ct_obs: {self.ct_obs}')
        print(f'nn_mode: {self.nn_mode}')
        print(f'nn_para: {self.nn_para}')
        print(f'count_rep: {self.count_rep}')
        print(f'include_self: {self.include_self}')
        print(f'n_scales: {self.n_scales}')
        
    def estimate_radius(self):
        spatialmat = self.adata.obsm['spatial']
        min_dist_list = []
        cur_distmat = squareform(pdist(spatialmat))
        np.fill_diagonal(cur_distmat, np.inf)
        cur_min_dist = np.min(cur_distmat, axis=0)
        min_dist_list.append(cur_min_dist)
        min_dist_array = np.hstack(min_dist_list)
        neighbor_sz = np.median(min_dist_array)
        print(f'estimated radius: {neighbor_sz}')
        self.estimated_radius = neighbor_sz
        
    def mp_helper(self, cur_scale, track_nn_time=False, track_nn_memory=False):
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
            cur_neighbors_cls = cls_array[cur_neighbors]
            cur_cls_unique, cur_cls_count = np.unique(cur_neighbors_cls, return_counts=1)
            cur_cls_idx = [np.where(ME_var_names_np_unique == c)[0][0] for c in cur_cls_unique]
            ME_X[i, cur_cls_idx] = cur_cls_count
        cur_ME_key = f'scale{cur_scale}'
        cur_X = ME_X
        return cur_ME_key, cur_X, nn_time, nn_memory
        
    def k_representation_mp(self, mp=200, track_dim_time=False, track_dim_memory=False, track_nn_time=False, track_nn_memory=False):
        from multiprocessing import Process, Pool
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
                cur_neighbors_cls = cls_array[cur_neighbors]
                cur_cls_unique, cur_cls_count = np.unique(cur_neighbors_cls, return_counts=1)
                cur_cls_idx = [np.where(ME_var_names_np_unique == c)[0][0] for c in cur_cls_unique]
                ME_X[i, cur_cls_idx] = cur_cls_count
            cur_ME_key = f'scale{cur_scale}'
            if self.count_rep == 'a':
                cur_X = ME_X
            elif self.count_rep == 's':
                cur_X = ME_X - ME_X_prev
                ME_X_prev = ME_X
            print(f'scale {cur_scale}, median #cells per ring (r={self.nn_para}):', np.median(np.sum(cur_X, axis=1)))
            self.adata.obsm[cur_ME_key] = cur_X.copy()
            if group_norm:
                self.adata.obsm[cur_ME_key] = self.adata.obsm[cur_ME_key] / np.sum(self.adata.obsm[cur_ME_key], axis=1, keepdims=True)
                self.adata.obsm[cur_ME_key] = np.nan_to_num(self.adata.obsm[cur_ME_key], 0)
        self.generate_ct_representation()
        dim_time, dim_memory = self.preprocess_adata_MENDER(track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        return None, dim_time, dim_memory, nn_time, nn_memory
            
    def radius_representation(self, group_norm=False, track_dim_time=False, track_dim_memory=False, track_nn_time=False, track_nn_memory=False):
        cls_array = self.ct_array
        ME_var_names_np_unique = self.ct_unique
        ME_X_prev = np.zeros(shape=(cls_array.shape[0], ME_var_names_np_unique.shape[0]))
        nn_time = 0.0
        nn_memory = 0.0
        for i in range(self.n_scales):
            cur_scale = i
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
                cur_neighbors_cls = cls_array[cur_neighbors]
                cur_cls_unique, cur_cls_count = np.unique(cur_neighbors_cls, return_counts=1)
                cur_cls_idx = [np.where(ME_var_names_np_unique == c)[0][0] for c in cur_cls_unique]
                ME_X[i, cur_cls_idx] = cur_cls_count
            cur_ME_key = f'scale{cur_scale}'
            if self.count_rep == 'a':
                cur_X = ME_X
            elif self.count_rep == 's':
                cur_X = ME_X - ME_X_prev
                ME_X_prev = ME_X
            print(f'scale {cur_scale}, median #cells per radius (r={self.nn_para}):', np.median(np.sum(cur_X, axis=1)))
            self.adata.obsm[cur_ME_key] = cur_X.copy()
            if group_norm:
                self.adata.obsm[cur_ME_key] = self.adata.obsm[cur_ME_key] / np.sum(self.adata.obsm[cur_ME_key], axis=1, keepdims=True)
                self.adata.obsm[cur_ME_key] = np.nan_to_num(self.adata.obsm[cur_ME_key], 0)
        self.generate_ct_representation()
        dim_time, dim_memory = self.preprocess_adata_MENDER(track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        return None, dim_time, dim_memory, nn_time, nn_memory
        
    def k_representation(self, track_dim_time=False, track_dim_memory=False, track_nn_time=False, track_nn_memory=False):
        cls_array = self.ct_array
        ME_var_names_np_unique = self.ct_unique
        ME_X_prev = np.zeros(shape=(cls_array.shape[0], ME_var_names_np_unique.shape[0]))
        nn_time = 0.0
        nn_memory = 0.0
        for i in range(self.n_scales):
            cur_scale = i
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
                cur_neighbors_cls = cls_array[cur_neighbors]
                cur_cls_unique, cur_cls_count = np.unique(cur_neighbors_cls, return_counts=1)
                cur_cls_idx = [np.where(ME_var_names_np_unique == c)[0][0] for c in cur_cls_unique]
                ME_X[i, cur_cls_idx] = cur_cls_count
            cur_ME_key = f'scale{cur_scale}'
            if self.count_rep == 'a':
                cur_X = ME_X
            elif self.count_rep == 's':
                cur_X = ME_X - ME_X_prev
                ME_X_prev = ME_X
            self.adata.obsm[cur_ME_key] = cur_X
        self.generate_ct_representation()
        dim_time, dim_memory = self.preprocess_adata_MENDER(track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        return None, dim_time, dim_memory, nn_time, nn_memory
        
    def generate_ct_representation(self):
        ME_var_names_np_unique = self.ct_unique
        whole_feature_list = []
        while_feature_X = []
        for ct_idx in range(len(ME_var_names_np_unique)):
            rep_list = []
            for i in range(self.n_scales):
                rep_list.append(self.adata.obsm[f'scale{i}'][:, ct_idx])
                whole_feature_list.append(f'ct{ct_idx}scale{i}')
                while_feature_X.append(self.adata.obsm[f'scale{i}'][:, ct_idx])
            cur_ct_rep = np.array(rep_list).transpose()
            cur_obsm = f'ct{ct_idx}'
            self.adata.obsm[cur_obsm] = cur_ct_rep
        self.adata.obsm[f'whole'] = np.array(while_feature_X).transpose()
        adata_feature = ad.AnnData(X=np.array(while_feature_X).transpose())
        adata_feature.obs_names = self.adata.obs_names
        adata_feature.var_names = whole_feature_list
        adata_feature.obsm['spatial'] = self.adata.obsm['spatial']
        for k in self.adata.obs.keys():
            adata_feature.obs[k] = self.adata.obs[k]
        if 'spatial' in self.adata.uns:
            adata_feature.uns['spatial'] = self.adata.uns['spatial']
        self.adata_MENDER = adata_feature
        
    def preprocess_adata_MENDER(self, mode=0, neighbor=True, track_dim_time=False, track_dim_memory=False):
        dim_time = 0.0
        dim_memory = 0.0
        if not hasattr(self, 'is_adata_MENDER_preprocess') or not self.is_adata_MENDER_preprocess:
            if mode == 0:
                sc.pp.normalize_total(self.adata_MENDER)
                sc.pp.log1p(self.adata_MENDER)
            elif mode == 1:
                sc.pp.normalize_total(self.adata_MENDER)
            elif mode == 2:
                sc.pp.log1p(self.adata_MENDER)
            elif mode == 3:
                pass
            if track_dim_time:
                dim_start_time = time.time()
            if track_dim_memory:
                dim_start_memory = psutil.Process().memory_info().rss / (1024 ** 2)
            if self.dim_reduction == 'pca':
                sc.pp.pca(self.adata_MENDER)
                self.adata_MENDER.obsm['X_dim_reduction'] = self.adata_MENDER.obsm['X_pca'].copy()
            elif self.dim_reduction == 'nmf':
                if np.any(self.adata_MENDER.X < 0):
                    raise ValueError("NMF requires non-negative input. Use mode=1 or mode=3.")
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
            if track_dim_time:
                dim_time = time.time() - dim_start_time
            if track_dim_memory:
                current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
                dim_memory = max(current_memory - dim_start_memory, 0)
            if neighbor:
                sc.pp.neighbors(self.adata_MENDER, use_rep='X_dim_reduction', random_state=self.random_seed)
            self.is_adata_MENDER_preprocess = True
        return dim_time, dim_memory
    
    def run_clustering_normal(self, target_k, run_umap=False, if_reprocess=True, track_dim_time=False, track_dim_memory=False):
        dim_time = 0.0
        dim_memory = 0.0
        if if_reprocess:
            dim_time, dim_memory = self.preprocess_adata_MENDER(mode=0, neighbor=True, track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        if run_umap:
            sc.tl.umap(self.adata_MENDER, obsm_key='X_dim_reduction')
            self.adata_MENDER.obsm['X_MENDERMAP2D'] = self.adata_MENDER.obsm['X_umap'].copy()
        if target_k > 0:
            res = res_search(self.adata_MENDER, target_k=target_k, random_state=self.random_seed, use_rep='X_dim_reduction')
            sc.tl.leiden(self.adata_MENDER, resolution=res, key_added=f'MENDER_leiden_k{target_k}', random_state=self.random_seed)
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_leiden_k{target_k}'].copy()
        elif target_k < 0:
            res = -target_k
            sc.tl.leiden(self.adata_MENDER, resolution=res, key_added=f'MENDER_leiden_res{res}', random_state=self.random_seed)
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_leiden_res{res}'].copy()
        else:
            print('please input a valid target_k')
            return 0.0, 0.0
        return dim_time, dim_memory
    
    def run_clustering_mclust(self, target_k, run_umap=False, track_dim_time=False, track_dim_memory=False):
        dim_time = 0.0
        dim_memory = 0.0
        if run_umap:
            dim_time, dim_memory = self.preprocess_adata_MENDER(mode=0, neighbor=True, track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
            sc.tl.umap(self.adata_MENDER, obsm_key='X_dim_reduction')
            self.adata_MENDER.obsm['X_MENDERMAP2D'] = self.adata_MENDER.obsm['X_umap'].copy()
        else:
            dim_time, dim_memory = self.preprocess_adata_MENDER(mode=0, neighbor=False, track_dim_time=track_dim_time, track_dim_memory=track_dim_memory)
        if target_k > 0:
            self.adata_MENDER = STAGATE.mclust_R(self.adata_MENDER, used_obsm='X_dim_reduction', num_cluster=target_k)
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_mclust_k{target_k}'].copy()
        else:
            print('please input a valid target_k')
            return 0.0, 0.0
        return dim_time, dim_memory
    
    def output_cluster(self, dirname, obs):
        sc.pl.embedding(self.adata_MENDER, basis='X_MENDERMAP2D', color=obs, save=f'_umap_{obs}')
        path = dirname
        os.mkdir(f'figures/spatial_{path}')
        adata_feature = self.adata_MENDER
        for i in range(len(self.batch_list)):
            cur_batch = self.batch_list[i]
            cur_a = adata_feature[adata_feature.obs[self.batch_obs] == cur_batch]
            ax = sc.pl.embedding(cur_a, basis='spatial', color=obs, show=False, title=cur_batch, save=None)
            ax.axis('equal')
            plt.savefig(f'figures/spatial_{path}/{cur_batch}.png', dpi=200, bbox_inches='tight', transparent=True)
            plt.close()

class SMENDER(object):
    def __init__(self, adata, batch_obs, ct_obs='ct', verbose=0, random_seed=666, ann=None, dim_reduction='pca'):
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
        self.adata_list = []
        self.batch_list = np.array(self.adata.obs[self.batch_obs].cat.categories)
        for b in self.batch_list:
            cur_a = self.adata[self.adata.obs[self.batch_obs] == b]
            self.adata_list.append(cur_a)
            
    def set_ct_obs(self, new_ct):
        if new_ct not in self.adata.obs:
            print('Please input a valid cell type obs')
            return
        else:
            self.ct_obs = new_ct
            self.ct_unique = np.array(self.adata.obs[self.ct_obs].cat.categories)
    
    def set_MENDER_para(self, nn_mode='k', nn_para=1, count_rep='s', include_self=True, n_scales=15):
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
            cur_MENDER.set_MENDER_para(
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
        cur_MENDER.set_MENDER_para(
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
        for i in range(len(self.batch_list)):
            cur_batch_name = self.batch_list[i]
            cur_batch_adata = self.adata_list[i]
            cur_MENDER = SMENDER_single(cur_batch_adata, ct_obs=self.ct_obs, verbose=self.verbose, random_seed=self.random_seed)
            cur_MENDER.estimate_radius()
            print(f'{cur_batch_name}: estimated radius: {cur_MENDER.estimated_radius}')
        
    def run_representation_mp(self, mp=200, group_norm=False):
        print('default number of processes is 200')
        from multiprocessing import Pool
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
        self.adata_MENDER = adata_MENDER_list[0].concatenate(adata_MENDER_list[1:])
        self.adata_MENDER_dump = self.adata_MENDER.copy()
        if self.is_tracking_dim_reduction_time:
            self.dim_reduction_time += total_dim_time
        if self.is_tracking_dim_reduction_memory:
            self.dim_reduction_memory += total_dim_memory
        if self.is_tracking_nn_time:
            self.nn_time += total_nn_time
        if self.is_tracking_nn_memory:
            self.nn_memory += total_nn_memory
    
    def refresh_adata_MENDER(self):
        del self.adata_MENDER
        self.adata_MENDER = self.adata_MENDER_dump.copy()
        self.is_adata_MENDER_preprocess = False
        
    def preprocess_adata_MENDER(self, neighbor=True):
        dim_time, dim_memory = SMENDER_single.preprocess_adata_MENDER(
            self, mode=0, neighbor=neighbor,
            track_dim_time=self.is_tracking_dim_reduction_time,
            track_dim_memory=self.is_tracking_dim_reduction_memory
        )
        if self.is_tracking_dim_reduction_time:
            self.dim_reduction_time += dim_time
        if self.is_tracking_dim_reduction_memory:
            self.dim_reduction_memory += dim_memory
        
    def run_clustering_normal(self, target_k, run_umap=False, if_reprocess=True):
        dim_time, dim_memory = SMENDER_single.run_clustering_normal(
            self, target_k, run_umap=run_umap, if_reprocess=if_reprocess,
            track_dim_time=self.is_tracking_dim_reduction_time,
            track_dim_memory=self.is_tracking_dim_reduction_memory
        )
        if self.is_tracking_dim_reduction_time:
            self.dim_reduction_time += dim_time
        if self.is_tracking_dim_reduction_memory:
            self.dim_reduction_memory += dim_memory
        
    def run_clustering_mclust(self, target_k, run_umap=False):
        dim_time, dim_memory = SMENDER_single.run_clustering_mclust(
            self, target_k, run_umap=run_umap,
            track_dim_time=self.is_tracking_dim_reduction_time,
            track_dim_memory=self.is_tracking_dim_reduction_memory
        )
        if self.is_tracking_dim_reduction_time:
            self.dim_reduction_time += dim_time
        if self.is_tracking_dim_reduction_memory:
            self.dim_reduction_memory += dim_memory
        
    def output_cluster(self, dirname, obs):
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
        cur_batch = self.batch_list[idx]
        adata_feature = self.adata_MENDER
        cur_a = adata_feature[adata_feature.obs[self.batch_obs] == cur_batch]
        sc.pl.embedding(cur_a, basis='spatial', color=obs, show=True, title=cur_batch)
        
    def output_cluster_all(self, obs='MENDER', obs_gt='gt'):
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
            plt.savefig(f'figures/spatial_{path}/{cur_batch}.png', dpi=200, bbox_inches='tight', transparent=True)
            plt.close()
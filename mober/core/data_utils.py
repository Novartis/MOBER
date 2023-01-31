import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import os

from scipy.sparse import csr_matrix

from sklearn.utils.class_weight import compute_class_weight

# modified from https://discuss.pytorch.org/t/sparse-dataset-and-dataloader/55466, credits to ironv
class SparseDataset(Dataset):
    def __init__(self, sp_matrix, label, device='cpu'):
        if type(sp_matrix) != csr_matrix: csr = csr_matrix(sp_matrix)
        else: csr = sp_matrix
        
        self.dim = csr.shape
        self.device = torch.device(device)

        self.indptr = torch.tensor(csr.indptr, dtype=torch.int64, device=self.device)
        self.indices = torch.tensor(csr.indices, dtype=torch.int64, device=self.device)
        self.data = torch.tensor(csr.data, dtype=torch.float32, device=self.device)

        self.label = torch.tensor(label, dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.dim[0]

    def __getitem__(self, idx):
        obs = torch.zeros((self.dim[1],), dtype=torch.float32, device=self.device)
        ind1,ind2 = self.indptr[idx],self.indptr[idx+1]
        obs[self.indices[ind1:ind2]] = self.data[ind1:ind2]

        return obs,self.label[idx]
    
def get_class_weights(class_series, balanced_sources):
    sorted_classes = sorted(class_series.unique())
    source = class_series.astype(pd.CategoricalDtype(sorted_classes, ordered=True))
    src_weight_factors = np.ones(source.unique().shape)
    if balanced_sources:
        src_weight_factors = compute_class_weight("balanced", classes=sorted_classes, y=source)
    return src_weight_factors


def create_dataloaders_from_adata(adata, batch_size, val_set_size, random_seed, use_sparse_mat=False):
    
    assert val_set_size >= 0 and val_set_size < 1.0
    samples = adata.obs.index.values
    splt = int(adata.shape[0]*(1-val_set_size))
    np.random.seed(random_seed)
    sample_inds = np.arange(len(samples))
    np.random.shuffle(sample_inds)
    np.random.seed() # reset seed
    
    tr_samples  = samples[sample_inds[:splt]]
    val_samples = samples[sample_inds[splt:]]
    
    label_encode = pd.get_dummies(sorted(adata.obs.data_source.unique()))
    label = pd.get_dummies(adata.obs.data_source)
    
    if use_sparse_mat:
        train_data = SparseDataset(adata[tr_samples,:].X,label.loc[tr_samples,:].values)
        if len(val_samples)>0:
            val_data   = SparseDataset(adata[val_samples,:].X,label.loc[val_samples,:].values)
    else:
        try: adata.X = adata.X.todense()
        except: None
        train_data = TensorDataset(torch.Tensor(adata[tr_samples,:].X) ,torch.Tensor(label.loc[tr_samples,:].values))
        if len(val_samples) > 0:
            val_data   = TensorDataset(torch.Tensor(adata[val_samples,:].X),torch.Tensor(label.loc[val_samples,:].values))
        
    
    train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
    if len(val_samples) > 0: val_loader   = DataLoader(val_data,  batch_size=batch_size, shuffle=True)
    else: val_loader = None
    
    return train_loader, val_loader, label_encode
    
    

    
    
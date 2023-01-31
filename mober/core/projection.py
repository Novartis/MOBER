import os
import torch

import numpy as np
import pandas as pd

from mober.core import data_utils
import scanpy as sc
from scipy.sparse import  csr_matrix 

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from mober.models import utils as model_utlis
from mober.models.batch_vae import BatchVAE

def decode(data_loader, model, device,decimals):
    """
    Get decodings numpy array from a data loader of data and a trained model

    :param data_loader: data loader that returns data and batch (source) annotations
    :param model: trained model
    :param device: device - cpu or gpu
    :return: a tuple of numpy arrays, one of decodings and another with encodings
    """
    decoded = []
    encoded = []
    model.eval()
    with torch.no_grad():
        for data, batch in data_loader:
            data = data.to(device)
            batch = batch.to(device)
            dec, enc = model(data, batch)[:2]
            encoded.append(enc)
            decoded.append(dec)  
    
    encoded = torch.cat(encoded,dim=0).detach().cpu().numpy().round(decimals=decimals)
    decoded = torch.cat(decoded,dim=0).detach().cpu().numpy().round(decimals=decimals)
    
    return decoded, encoded

def load_model(model_dir, device):
    features = pd.read_csv(os.path.join(model_dir,'features.csv'),index_col=0).index
    label_encode = pd.read_csv(os.path.join(model_dir,'label_encode.csv'),index_col=0)
    params = pd.read_csv(os.path.join(model_dir,'params.csv'),index_col=0)
    
    model, _ = model_utlis.create_model(BatchVAE, 
                                         device, 
                                         features.shape[0], 
                                         int(params.loc['encoding_dim','value']),  
                                         label_encode.shape[0],  
                                         filename=os.path.join(model_dir,'batch_ae_final.model'))
    return model, features, label_encode

def do_projection(model,adata, onto, label_encode, device, decimals=4, batch_size=1600, use_sparse_mat=False):

    label = np.array([label_encode[onto].values for _ in range(adata.shape[0])])
    
    if use_sparse_mat: dataset = data_utils.SparseDataset(adata.X,label)
    else: 
        try: X = adata.X.todense()
        except: X = adata.X
        dataset = TensorDataset(torch.Tensor(X),torch.Tensor(label))
        
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    
    projected, z = decode(data_loader, model, device, decimals)
    
    if use_sparse_mat: proj_adata = sc.AnnData(csr_matrix(projected),obs=adata.obs,var=adata.var)
    else:  proj_adata = sc.AnnData(projected,obs=adata.obs,var=adata.var)
    proj_adata.obs['projected_onto'] = onto
    z_adata    = sc.AnnData(z, obs=adata.obs,var=pd.DataFrame(index=[f'z_{i}' for i in range(z.shape[1])])) 
    
    return proj_adata, z_adata
    
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = sc.read(args.projection_file)
    model, features, label_encode = load_model(args.model_dir, device)
    adata = adata[:,features]
    
    proj_adata, z_adata = do_projection(model, adata, args.onto, label_encode, device, decimals=args.decimals, batch_size=1600)
    proj_adata.write(args.output_file)
    
    
    
    
    
    
    
    
 
    
    
    
    
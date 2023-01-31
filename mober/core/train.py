import os
import time
import torch

from mober.core import utils
from mober.core import data_utils
from mober.models import utils as model_utils

import argparse
import copy

import pandas as pd
import numpy as np
import numexpr
import mlflow

from mober.models.batch_vae import BatchVAE
from mober.models.mlp import MLP

from mober.loss.classification import loss_function_classification
from mober.loss.vae import loss_function_vae

import scanpy as sc

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
def validation(model_BatchAE,model_src_adv,val_loader,device, args, log, src_weights_src_adv, epoch):
    model_BatchAE.eval()
    model_src_adv.eval()
    
    epoch_ae_loss_val      = 0.0
    epoch_src_adv_loss_val = 0.0
    epoch_tot_loss_val     = 0.0
    
    with torch.no_grad():
        for data, batch in val_loader:

            data = data.to(device)
            batch = batch.to(device)
            
            dec, enc, means, stdev = model_BatchAE(data, batch)
            v_loss = loss_function_vae(dec, data, means, stdev, kl_weight=args.kl_weight)
         
            src_pred = model_src_adv(enc)
            loss_src_adv = loss_function_classification(src_pred, batch, src_weights_src_adv)
            loss_ae = v_loss - args.src_adv_weight * loss_src_adv

            epoch_ae_loss_val += v_loss.detach().item()
            epoch_src_adv_loss_val += loss_src_adv.detach().item()
            epoch_tot_loss_val += loss_ae.detach().item()

    log.log_metric("val_loss_ae" , epoch_ae_loss_val      / len(val_loader.dataset), epoch)
    log.log_metric("val_loss_adv", epoch_src_adv_loss_val / len(val_loader.dataset), epoch)
    log.log_metric("val_loss_tot", epoch_tot_loss_val     / len(val_loader.dataset), epoch)
    
    return epoch_ae_loss_val

def train_model(model_BatchAE, 
                optimizer_BatchAE, 
                model_src_adv, 
                optimizer_src_adv,
                train_loader, 
                val_loader, 
                src_weights_src_adv,
                run_dir, 
                device,
                log,
                args):
    
    
    # Early stopping settings
    best_model_loss = np.inf
    waited_epochs = 0
    early_stop = False
    
    ae_model_file  =  os.path.join(run_dir, "models", "batch_ae_final.model")
    src_model_file =  os.path.join(run_dir, "models", "src_adv_final.model")
                                  
    for epoch in range(args.epochs):
        if early_stop: break
        
        epoch_ae_loss      = 0.0
        epoch_src_adv_loss = 0.0
        epoch_tot_loss     = 0.0 
        
        model_BatchAE.train()
        model_src_adv.train()
        for data, batch in train_loader:
            data = data.to(device)
            batch = batch.to(device)

            dec, enc, means, stdev = model_BatchAE(data, batch)
            v_loss = loss_function_vae(dec, data, means, stdev, kl_weight=args.kl_weight)

            # Source adversary
            model_src_adv.zero_grad()

            src_pred = model_src_adv(enc)

            loss_src_adv = loss_function_classification(src_pred, batch, src_weights_src_adv)
            loss_src_adv.backward(retain_graph=True)
            epoch_src_adv_loss += loss_src_adv.detach().item()
            optimizer_src_adv.step()

            src_pred = model_src_adv(enc)
            loss_src_adv = loss_function_classification(src_pred, batch, src_weights_src_adv)

            # Update ae
            model_BatchAE.zero_grad()
            loss_ae = v_loss - args.src_adv_weight * loss_src_adv
            loss_ae.backward()
            epoch_ae_loss += v_loss.detach().item()
            optimizer_BatchAE.step()
            
            epoch_tot_loss += loss_ae.detach().item()
            
        log.log_metric("train_loss_ae" , epoch_ae_loss      / len(train_loader.dataset), epoch)
        log.log_metric("train_loss_adv", epoch_src_adv_loss / len(train_loader.dataset), epoch)
        log.log_metric("train_loss_tot", epoch_tot_loss     / len(train_loader.dataset), epoch)
        
        
        # Validation
        if args.val_set_size != 0:
            epoch_ae_loss_val = validation(model_BatchAE,model_src_adv,val_loader,device, args, log, src_weights_src_adv,epoch)

            # Early stop
            if epoch_ae_loss_val < best_model_loss: # there is an improvement, update the best_val_loss and save the model
                best_model_loss = epoch_ae_loss_val
                waited_epochs = 0
                model_utils.save_model(model_BatchAE, optimizer_BatchAE, epoch, epoch_ae_loss/len(train_loader.dataset)     ,ae_model_file , device)
                model_utils.save_model(model_src_adv, optimizer_src_adv, epoch, epoch_src_adv_loss/len(train_loader.dataset),src_model_file, device)

            else:
                waited_epochs += 1
                if waited_epochs > args.patience: early_stop = True
                
    if args.val_set_size == 0: 
        model_utils.save_model(model_BatchAE, optimizer_BatchAE, epoch, epoch_ae_loss/len(train_loader.dataset)     ,ae_model_file , device)
        model_utils.save_model(model_src_adv, optimizer_src_adv, epoch, epoch_src_adv_loss/len(train_loader.dataset),src_model_file, device)


def main(args):
    if args.use_mlflow:
        run_dir = os.path.join(args.tmp_dir,str(int(time.time())))
        mlflow.set_tracking_uri(args.mlflow_storage_path)
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run(run_name=args.run_name)
    else: 
        run_dir = args.output_dir
    
    utils.create_temp_dirs(run_dir)
    
    
    log = utils.log_obj(args.use_mlflow,run_dir)
    log.log_params(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    numexpr.set_num_threads(numexpr.detect_number_of_cores())
    
    adata = sc.read(args.train_file)

    train_loader, val_loader, label_encode = data_utils.create_dataloaders_from_adata(adata, 
                                                                                     args.batch_size, 
                                                                                     args.val_set_size, 
                                                                                     args.random_seed,
                                                                                     args.use_sparse_mat
                                                                                     )
    # save features and label encoding
    features = adata.var.index.to_frame()
    label_encode.to_csv(os.path.join(run_dir, 'models', 'label_encode.csv'))
    features.to_csv(os.path.join(run_dir, 'models', 'features.csv'))
    
    
    set_seed(args.random_seed)
    

    model_BatchAE, optimizer_BatchAE = model_utils.create_model(BatchVAE,
                                                                device,
                                                                features.shape[0],
                                                                args.encoding_dim,
                                                                label_encode.shape[0],
                                                                lr=args.batch_ae_lr,
                                                                filename=None)

    model_src_adv, optimizer_src_adv = model_utils.create_model(MLP,
                                                                device,
                                                                args.encoding_dim,
                                                                label_encode.shape[0],
                                                                lr=args.src_adv_lr,
                                                                filename=None)

    
    
    src_weights_src_adv = torch.tensor(
        data_utils.get_class_weights(adata.obs.data_source, args.balanced_sources_src_adv), dtype=torch.float).to(device)
    
    
    train_model(model_BatchAE, 
                optimizer_BatchAE, 
                model_src_adv, 
                optimizer_src_adv,
                train_loader, 
                val_loader, 
                src_weights_src_adv,
                run_dir,
                device,
                log,
                args)
    
    log.end_log()
    

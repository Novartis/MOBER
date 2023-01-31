import sys
import argparse

def main():
    parser = argparse.ArgumentParser(prog='mober', 
                                     description='''MOBER is a deep learning method that allows for integration of \
                                     cancer models (CCLEs and PTXs) that are closest to patient tumors of interest based on \
                                     mRNA expression data, without relying on annotated disease labels. It projects one dataset \
                                     onto another and transforms the transcriptional profiles of cancer models (CCLEs and PTXs) \
                                     into TCGA patient tumors.''')
    
    subparsers = parser.add_subparsers(dest='mode')
    
    
    ###################### Train Mode #######################
    tparser = subparsers.add_parser('train',help='Train MOBER')
    
    tparser.add_argument(
        "--train_file",
        metavar = '',
        help = "A h5ad file that contains all the samples as well as a 'data_source' column."
    )
    
    tparser.add_argument(
        "--use_sparse_mat",
        action = "store_true",
        help = "If to use sparse dataloader. Can be used when the training dataset is huge. Default False"
    )
    
    tparser.add_argument(
        "--src_adv_weight",
        type = float,
        metavar = '',
        default=0.01,
        help = "Weight of the source adversary loss. Default 0.01",
    )
    
    tparser.add_argument(
        "--src_adv_lr",
        type = float,
        metavar = '',
        help = "Learning rate. Default 1e-3",
        default=1e-3
    )
    
    tparser.add_argument(
        "--batch_ae_lr",
        type = float,
        metavar = '',
        help="Learning rate. Default 1e-3",
        default=1e-3
    )
    
    tparser.add_argument(
        "--val_set_size",
        type = float,
        metavar = '',
        help = "Fraction of samples that constitute the validation set. Default 0.0",
        default = 0.0
    )

    tparser.add_argument(
        "--encoding_dim",
        type = int,
        default= 64,
        metavar = '',
        help = "Size of the embeddings. Default 64",
    )
    
    tparser.add_argument(
        "--balanced_sources_ae",
        action = "store_true",
        help = "Flag that enables sample weights to balance according to the source in ae loss. Default False"
    )
    
    tparser.add_argument(
        "--balanced_sources_src_adv",
        action = "store_true",
        help = "Flag that enables sample weights to balance according to the source in source adversary loss. Default False"
    )
    
    tparser.add_argument(
        "--batch_size",
        type = int,
        metavar = '',
        default = 1600,
        help = 'Default 1600'
    )
    
    tparser.add_argument(
        "--epochs",
        type = int,
        metavar = '',
        default = 3000,
        help = 'Max number of training epochs, Default 15000. Eearly Stopping implemented.'
    )
    tparser.add_argument(
        "--random_seed",
        type = int,
        default=100,
        metavar = '',
        help = 'Default 100'
    )
    
    
    tparser.add_argument(
        "--kl_weight",
        type = float,
        default=1e-5,
        metavar = '',
        help = 'Default 1e-6. Weight for KL loss.'
    )
    
    tparser.add_argument(
        "--patience",
        type = int,
        default = 100,
        metavar = '',
        help = "Number of patience epochs for early stopping. Default 100"
    )
    
    tparser.add_argument(
        "--output_dir",
        type = str,
        default = None,
        metavar = '',
        help="Output path in case MLflow not used.'"
    )
    
    
    ##### MLFlow arguments####
    tparser.add_argument(
        "--use_mlflow",
        action = "store_true",
        help = "Used if all results to be tracked by MLflow. Default False"
    )
    
    tparser.add_argument(
        "--mlflow_storage_path",
        type = str,
        metavar = '',
        default = 'http://nrchbs-ldl31318.nibr.novartis.net:5000',
        help = 'Default: http://nrchbs-ldl31318.nibr.novartis.net:5000'
    )
    
    tparser.add_argument(
        "--experiment_name",
        type=str,
        default = "mober",
        metavar = '',
        help ='Expriment name for MLFlow. Default mober'
    )
    
    tparser.add_argument(
        "--run_name",
        type = str,
        default = "run",
        metavar = '',
        help = "Run name for MLFlow. Default run"
    )
    
    tparser.add_argument(
        "--tmp_dir",
        type = str,
        default = "tmp",
        metavar = '',
        help = "Temporary directory for MLflow. Default ./tmp"
    )
    
    
    
    
    
    ###################### Projection Mode #######################
    pparser = subparsers.add_parser('projection',help = 'Make projection')
    
    pparser.add_argument(
        "--model_dir",
        type = str,
        metavar = '',
        help = "Path to model file",
    )
    
    pparser.add_argument(
        "--onto",
        type = str,
        metavar = '',
        help = "The target 'data_source' ID which all samples will be projected onto",
    )
    
    pparser.add_argument(
        "--projection_file",
        type = str,
        metavar = '',
        help = "An input file that contains all the gene expression of all samples. h5ad format"
    )
    
    pparser.add_argument(
        "--output_file",
        type = str,
        metavar = '',
        help = "Name for output h5ad file for projected values",
    )
    
    pparser.add_argument(
        "--decimals",
        type = int,
        default=4,
        metavar = '',
        help = "Floating-point numbers for the output file. Default 4",
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        from mober.core import train
        train.main(args)
        
    if args.mode == 'projection':
        from mober.core import projection
        projection.main(args)
    

if __name__ == "__main__": 
    main()
    

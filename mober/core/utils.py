import os
import pandas as pd
import shutil
from pathlib import Path
import mlflow


def create_temp_dirs(tmp_dir):
    Path(os.path.join(tmp_dir, "models")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(tmp_dir, "metrics")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(tmp_dir, "projection")).mkdir(parents=True, exist_ok=True)


def remove_temp_dirs(tmp_dir):
    shutil.rmtree(tmp_dir)
    
    


class log_obj:
    def __init__(self, use_mlflow, run_dir):
        self.use_mlflow = use_mlflow
        self.run_dir = run_dir
        self.fhands = {}
    
    def log_params(self,args):
        if self.use_mlflow: mlflow.log_params(vars(args))
        dfparams = pd.DataFrame(data=vars(args),index=['value']).transpose()
        dfparams.to_csv(os.path.join(self.run_dir, 'models', 'params.csv'))
        
    def log_metric(self,name,value,epoch):
        if self.use_mlflow: mlflow.log_metric(name, value, step=epoch)
        else:
            if name not in self.fhands.keys():
                fhand = open(os.path.join(self.run_dir,'metrics',name),'w',buffering=1)
                fhand.write('epoch\tvalue\n')
                self.fhands[name] = fhand
                
            self.fhands[name].write(f'{epoch}\t{value}\n')
    
    def end_log(self):
        if self.use_mlflow:
            mlflow.log_artifacts(self.run_dir)
            mlflow.end_run()
            remove_temp_dirs(self.run_dir)
        else: 
            for fhand in self.fhands.values(): fhand.close()
        
    
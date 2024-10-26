import json, os, sys, torch, argparse

import json, os, sys, torch, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
from shutil import copyfile, rmtree
from subprocess import check_output
from io import StringIO
from typing_extensions import override

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from model import CVAE
from loss import KLLoss
from annealer import Annealer, AnnealerStep

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocesses metabolomic data: discards columns with too many missing values, normalizes rows with row sum, imputes remaining missing values, normalizes rows with row sum again, standardizes columns and selects top columns based on least anova p-values')
    parser.add_argument("--train_feature_path", type=str,
                        help="train features in .tsv format after preprocessing, the first column name should be 'Sample' containing sample identifiers and the rest of the columns should contain metabolomic concentrations",
                        required=True, default=None)
    parser.add_argument("--train_condition_path", type=str,
                        help="train condiiton in .tsv format after preprocessing, the first column name should be 'Sample' containing sample identifiers, and the rest of the columns should each denote one disease group and contain either 0 or 1",
                        required=True, default=None)
    
    parser.add_argument("--val_feature_path", type=str,
                        help="validation features in .tsv format after preprocessing, the first column name should be 'Sample' containing sample identifiers and the rest of the columns should contain metabolomic concentrations",
                        required=True, default=None)
    parser.add_argument("--val_condition_path", type=str,
                        help="validation condiiton in .tsv format after preprocessing, the first column name should be 'Sample' containing sample identifiers, and the rest of the columns should each denote one disease group and contain either 0 or 1",
                        required=True, default=None)
    
    parser.add_argument("--model_name", type=str,
                        choices=['combined_hidden'],
                        help="name of the model architecture",
                        required=True, default=None)
    
    parser.add_argument("--syn_sample_count", type=int,
                        help="count of synthetic samples to generate",
                        required=True, default=None)
    
    parser.add_argument("--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

class CVAEDataset(Dataset):
    def __init__(self, feature_path, condition_path):
        self.feature_df = pd.read_csv(feature_path, sep='\t', index_col='Sample')
        self.condition_df = pd.read_csv(condition_path, sep='\t', index_col='Sample')
        
        print('self.feature_df', self.feature_df)
        print('self.condition_df', self.condition_df)
        
        self.feature_names = list(self.feature_df.columns)
        self.cohort_names = list(self.condition_df.columns)
        
        self.length = self.condition_df.shape[0]
        self.samples = list(self.condition_df.index)
        
        cohort_pct = np.sum(self.condition_df.to_numpy(), axis=0)
        cohort_pct = cohort_pct / np.sum(cohort_pct)
        self.cohort_pct = cohort_pct
        print('self.cohort_pct', self.cohort_pct)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        feature = self.feature_df.loc[sample, self.feature_names]
        feature = torch.from_numpy(feature.to_numpy()).float()
        
        condition = self.condition_df.loc[sample, self.cohort_names]
        condition = torch.from_numpy(condition.to_numpy()).float()
        return {
            'feature': feature,
            'condition': condition,
            'sample': sample
        }

class Synthesizer(pl.LightningModule):
    def __init__(self,
                 model_name,
                 feature_dim,
                 condition_dim,
                 hidden_dim,
                 latent_dim,
                 lr):
        super(Synthesizer, self).__init__()
        self.save_hyperparameters()
        model_map = {
            'combined_hidden': CVAE,
        }
        loss_map = {
            'kl': KLLoss,
            'mse': torch.nn.MSELoss,
        }
        self.model = model_map[model_name](feature_dim, condition_dim, hidden_dim, latent_dim)
        print('self.model', self.model)
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.kl_module = KLLoss()
        self.mse_module = torch.nn.MSELoss()
        self.annealer = Annealer(total_steps=10, cyclical=True, shape='linear')
        
        
    def sample_data_batch(self, cohort_pct, sample_count):       
        cohort_count = np.round(np.multiply(cohort_pct, sample_count)).astype(int)
        cohort = np.zeros(shape=(np.sum(cohort_count), cohort_count.size))
        past_samples = 0
        for cohort_i in range(cohort_count.size):
            cohort[past_samples: past_samples+cohort_count[cohort_i], cohort_i] = 1
            past_samples += cohort_count[cohort_i]
        condition = torch.from_numpy(cohort).float()
        condition = condition.to(self.device)
        z, samples = self.model.sample(condition)
        return condition, z, samples
    
    def sample_data(self, cohort_pct, sample_count):
        batch_max_size = 64
        batch_condition = []
        batch_z = []
        batch_samples = []
        
        batch_sizes = [batch_max_size] * (sample_count // batch_max_size) + [sample_count % batch_max_size]
        
        for batch_size in batch_sizes:
            condition, z, samples = self.sample_data_batch(cohort_pct, batch_size)
            batch_condition.append(condition)
            batch_z.append(z)
            batch_samples.append(samples)
            
        
        condition = torch.cat(batch_condition, 0)
        z = torch.cat(batch_z, 0)
        samples = torch.cat(batch_samples, 0)
        
        return condition, z, samples
                
    def forward(self, feature, condition):
        feature = feature.to(self.device)
        condition = condition.to(self.device)
        print('feature', feature.shape, 'condition', condition.shape)
        return self.model(feature, condition)
    
    def configure_optimizers(self):
        return Adam(self.parameters(),
                    lr=self.lr)
    
    def calc_loss(self,
                  pred_feat,
                  true_feat,
                  mean,
                  logvar,
                  prefix):
        
        print('calc_loss')
        print('pred_feat', pred_feat.shape, 'true_feat', true_feat.shape)
        print('mean', mean.shape, 'logvar', logvar.shape)
        
        print('pred_feat')
        print(pred_feat)
        print('true_feat')
        print(true_feat)
        print('mean')
        print(mean)
        print('logvar')
        print(logvar)
        
        loss = 0
        loss_dict = {}
        
        kl_loss = self.kl_module(mean, logvar)
        print('kl_loss', kl_loss, kl_loss.shape)
        loss_dict[prefix + '_kl_loss'] = kl_loss.item()
        
        if(abs(kl_loss.item()) > 0):
            slope, kl_loss = self.annealer(kl_loss/kl_loss.detach())
            loss_dict[prefix + '_slope'] = slope
        loss += kl_loss
        
        mse_loss = self.mse_module(pred_feat, true_feat)
        print('mse_loss', mse_loss, mse_loss.shape)
        loss += mse_loss/mse_loss.detach()
        loss_dict[prefix + '_mse_loss'] = mse_loss.item()
            
        loss_dict[prefix + '_loss'] = loss.item()
        
        print('loss_dict', loss_dict)
        self.log_dict(loss_dict, on_step=False, on_epoch=True)
        return loss
            
    
    def training_step(self, batch, batch_idx):
        print('training_step')
        z, mean, logvar, out = self.forward(batch['feature'], batch['condition'])
        loss = self.calc_loss(out, batch['feature'], mean, logvar, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        print('validation_step')
        z, mean, logvar, out = self.forward(batch['feature'], batch['condition'])
        loss = self.calc_loss(out, batch['feature'], mean, logvar, 'val')
        return loss
    
def plot_metric(train_x, val_x, train_y, val_y, xlabel, ylabel, title, png_path):
    plt.plot(train_x, train_y, label='Train')
    plt.plot(val_x, val_y, label='Val')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.legend()
    plt.savefig(png_path)
    plt.close()

def get_free_gpu():
    gpu_stats = check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_stats = gpu_stats.decode("utf-8")
    gpu_df = pd.read_csv(StringIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    gpu_df.to_csv('gpu_memory.tsv', sep='\t')
    gpu_df['memory.used'] = gpu_df['memory.used'].str.replace(" MiB", "").astype(int)
    gpu_df['memory.free'] = gpu_df['memory.free'].str.replace(" MiB", "").astype(int)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_id = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(gpu_id, gpu_df.iloc[gpu_id]['memory.free']))
    return gpu_id  

def generate_synthetic_data(train_feature_path,
                            train_condition_path,
                            val_feature_path,
                            val_condition_path,
                            model_name,
                            syn_sample_count):
    
    train_set = CVAEDataset(train_feature_path,
                            train_condition_path)
    val_set = CVAEDataset(val_feature_path,
                          val_condition_path)
    
    print('train_set', len(train_set),
          'val_set', len(val_set))
    
    data = train_set[0]
    feature_dim = data['feature'].shape[0]
    condition_dim = data['condition'].shape[0]
    
    print('feature_dim', feature_dim)
    print('condition_dim', condition_dim)
    
    hidden_dim = [4, 8]
    lr = [1e-4, 1e-5, 1e-6]
    batch_size = [128, 256]
    
    hparams = list(product(hidden_dim, lr, batch_size))
    hparam_label = ['hparam-'+str(i) for i in range(len(hparams))]
    
    log_dir = os.getcwd() + '/logs'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    hparam_df = pd.DataFrame(hparams,
                             columns=['hidden_dim', 'lr', 'batch_size'],
                             index=hparam_label)
    
    for i in range(len(hparam_label)):
        hparam_no = hparam_label[i]
        hparam = hparams[i]
        print(hparam_no, hparam)
        
        if(os.path.exists(hparam_no)):
            rmtree(hparam_no)
            
        Path(hparam_no).mkdir(parents=True)
        os.chdir(hparam_no)
        
        train_loader = DataLoader(train_set,
                                  batch_size=hparam[2],
                                  shuffle=True) # check shuffle in prediction
        val_loader = DataLoader(val_set,
                                batch_size=hparam[2],
                                shuffle=False)
        
        early_stopping = EarlyStopping(monitor='val_mse_loss',
                                       mode='min',
                                       patience=3,
                                       min_delta=1e-4,
                                       check_finite=True,
                                       stopping_threshold=0)
        
        annealer_step = AnnealerStep()
        
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              monitor='val_mse_loss',
                                              mode='min',
                                              auto_insert_metric_name=True)
        
        hparam_log_dir = log_dir + '/' + hparam_no
        if(os.path.exists(hparam_log_dir)):
            rmtree(hparam_log_dir)
        
        csv_logger = CSVLogger(hparam_log_dir, name=None)
        
        synthesizer = Synthesizer(model_name,
                                  feature_dim,
                                  condition_dim,
                                  hparam[0],
                                  int(hparam[0]//2),
                                  hparam[1])
        hparam_df.loc[hparam_no, 'param_count'] = sum(p.numel() for p in synthesizer.model.parameters())
        
        trainer = pl.Trainer(max_epochs=100, # try different values
                             callbacks=[early_stopping,
                                        checkpoint_callback,
                                        annealer_step],
                             log_every_n_steps=1,
                             accelerator="cuda",
                             devices=[get_free_gpu()],
                             enable_checkpointing=True,
                             logger=[csv_logger],
                             check_val_every_n_epoch=1,
                             val_check_interval=1.0,
                             enable_progress_bar=False,
                             num_sanity_val_steps=0)
        
        trainer.fit(synthesizer,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
        
        plot_dir = hparam_log_dir + '/plot'
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        metrics = ['loss', 'mse_loss', 'kl_loss']
        metric_path = hparam_log_dir + '/version_0/metrics.csv'
        metric_df = pd.read_csv(metric_path, sep=',')
        
        hparam_df.loc[hparam_no, 'last_epoch'] = metric_df['epoch'].max()
        
        for metric in metrics:
            print('Plotting', metric)
            train_metric = 'train_' + metric
            val_metric = 'val_' + metric
            print('metric_df', metric_df.columns)
            train = metric_df.dropna(subset=train_metric).sort_values(by='epoch')
            val = metric_df.dropna(subset=val_metric).sort_values(by='epoch')
            
            train_x = list(train['epoch'])
            train_y = list(train[train_metric])
            
            val_x = list(val['epoch'])
            val_y = list(val[val_metric])
            
            plot_metric(train_x,
                        val_x,
                        train_y,
                        val_y,
                        'Epoch',
                        metric,
                        metric+' across epochs',
                        plot_dir + '/' + metric + '.png')
    
        hparam_df.loc[hparam_no, 'val_mse_loss'] = metric_df['val_mse_loss'].min(skipna=True)
        
        best_model = Synthesizer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        best_model.eval()
        
        condition, z, out = best_model.sample_data(train_set.cohort_pct, syn_sample_count)
        condition = condition.cpu().detach().numpy()
        samples = ['sample-'+str(i) for i in range(1, condition.shape[0]+1)]
        condition_df = pd.DataFrame(condition,
                                   index=samples,
                                   columns=train_set.cohort_names)
        condition_df.index.name = 'Sample'
        condition_df.to_csv('synthetic_condition.tsv', sep='\t', index=True)
        
        z = z.cpu().detach().numpy()
        zdf = pd.DataFrame(z,
                           index=samples,
                           columns=['z'+str(i) for i in range(z.shape[1])])
        zdf.to_csv('z.tsv', sep='\t', index=True)
        
        out = out.cpu().detach().numpy()
        out_df = pd.DataFrame(out,
                              columns=train_set.feature_names,
                              index=samples)
        out_df.index.name = 'Sample'

        out_df.to_csv('synthetic_feature.tsv', sep='\t', index=True)        
        os.chdir('..')
        print('\n')
    hparam_df.to_csv(log_dir + '/hyperparameters.tsv', sep='\t', index=True)
    best_hparam = hparam_df['val_mse_loss'].idxmin()
    with open(log_dir + '/best_hparam.txt', 'w') as fp:
        fp.write(best_hparam)
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/train_cvae.log', 'w')
    
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    original_stderr = sys.stderr
    sys.stderr = log_file
    
    print(args)
    
    kwargs = vars(args)
    del kwargs['out_dir']
    generate_synthetic_data(**kwargs)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    log_file.flush()
    log_file.close()

if __name__ == "__main__":
    main(parse_args())

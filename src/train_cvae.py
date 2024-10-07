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
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from model import SeparateHiddenCVAE, CombinedHiddenCVAE
from loss import KLLoss, MultiLossEarlyStopping

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mic_path", type=str,
                        help="path to the train micriobiome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--train_met_path", type=str,
                        help="path to the train metabolome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--train_meta_path", type=str,
                        help="path to the train metadata data in .tsv format",
                        required=True, default=None)
    
    parser.add_argument("--val_mic_path", type=str,
                        help="path to the validation micriobiome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--val_met_path", type=str,
                        help="path to the validation metabolome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--val_meta_path", type=str,
                        help="path to the validation metadata data in .tsv format",
                        required=True, default=None)
    
    parser.add_argument("--model_name", type=str,
                        choices=['separate_hidden', 'combined_hidden'],
                        help="name of the synthetic data generation model",
                        required=True, default=None)
    parser.add_argument("--syn_sample_count", type=int,
                        help="count of synthetic samples to generate",
                        required=True, default=None)
    
    parser.add_argument("--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

class MicrobiomeMetabolomeDataset(Dataset):
    def __init__(self, microbiome_path, metabolome_path, condition_path):
        microbiome_df = pd.read_csv(microbiome_path, sep='\t', index_col='Sample')
        metabolome_df = pd.read_csv(metabolome_path, sep='\t', index_col='Sample')
        self.feature_df = pd.concat([microbiome_df, metabolome_df], axis=1) 
        self.condition_df = pd.read_csv(condition_path, sep='\t', index_col='Sample')
        
        self.feature_names = list(self.feature_df.columns)
        self.microbiome_names = list(microbiome_df.columns)
        self.metabolome_names = list(metabolome_df.columns)
        self.cohort_names = list(self.condition_df.columns)
        
        print('self.feature_names', self.feature_names)
        print('self.microbiome_names', self.microbiome_names)
        print('self.metabolome_names', self.metabolome_names)
        print('self.cohort_names', self.cohort_names)
        
        self.length = self.condition_df.shape[0]
        self.samples = list(self.condition_df.index)
        
        print('self.length', self.length)
        print('self.samples', self.samples)
        
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
            'separate_hidden': SeparateHiddenCVAE,
            'combined_hidden': CombinedHiddenCVAE,
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
        self.optim_loss = ['mse', 'kl']
        self.lr = lr
        self.loss_modules = {loss: loss_map[loss]() for loss in self.optim_loss}
        print('self.loss_modules', self.loss_modules)
        
    def sample_data(self, cohort_pct, sample_count):       
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
    
    def forward(self, feature, condition):
        feature = feature.to(self.device)
        condition = condition.to(self.device)
        print('feature', feature.shape, 'condition', condition.shape)
        return self.model(feature, condition)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def calc_loss(self, pred_feat, true_feat, mean, logvar, prefix):
        loss_dict = {}
        loss = 0
        
        kl_loss = self.loss_modules['kl'](mean, logvar)
        loss += kl_loss/kl_loss.detach()
        loss_dict[prefix + '_kl_loss'] = kl_loss.item()
        
        mse_loss = self.loss_modules['mse'](pred_feat, true_feat)
        loss += mse_loss/mse_loss.detach()
        loss_dict[prefix + '_mse_loss'] = mse_loss.item()
        
        '''l1_loss = [p.abs().sum() for p in self.model.parameters()]
        print('l1_loss', l1_loss)
        l1_loss = torch.mean(torch.Tensor(l1_loss))
        loss = l1_loss + loss
        loss_dict[prefix + '_l1_loss'] = l1_loss.item()'''
        
        loss_dict[prefix + '_loss'] = loss
        
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
    
    '''def predict_step(self, batch, batch_idx, dataloader_idx):
        print('predict_step')
        z, mean, logvar, out = self.forward(batch['feature'], batch['condition'])
        return batch['sample'], out'''
    
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

def generate_synthetic_data(train_mic_path,
                            train_met_path,
                            train_meta_path,
                            val_mic_path,
                            val_met_path,
                            val_meta_path,
                            model_name,
                            syn_sample_count):
    
    train_set = MicrobiomeMetabolomeDataset(train_mic_path,
                                            train_met_path,
                                            train_meta_path)
    val_set = MicrobiomeMetabolomeDataset(val_mic_path,
                                          val_met_path,
                                          val_meta_path)
    
    print('train_set', len(train_set),
          'val_set', len(val_set))
    
    data = train_set[0]
    feature_dim = data['feature'].shape[0]
    condition_dim = data['condition'].shape[0]
    
    print('feature_dim', feature_dim)
    print('condition_dim', condition_dim)
    
    hidden_dim = [feature_dim//2, feature_dim//4]
    lr = [1e-5, 1e-6]
    batch_size = [4, 8]
    
    hparams = list(product(hidden_dim, lr, batch_size))
    hparam_label = ['hparam-'+str(i) for i in range(len(hparams))]
    
    csv_log_dir = os.getcwd() + '/logs/csv_logs'
    Path(csv_log_dir).mkdir(parents=True, exist_ok=True)
    
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
        
        early_stopping = MultiLossEarlyStopping(monitor=['val_mse_loss', 'val_kl_loss'],
                                                mode=['min', 'min'],
                                                patience=[5, 5],
                                                min_delta=[0, 0])
        
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              monitor='val_mse_loss',
                                              mode='min',
                                              auto_insert_metric_name=True)
        
        hparam_csv_log_dir = csv_log_dir + '/' + hparam_no
        if(os.path.exists(hparam_csv_log_dir)):
            rmtree(hparam_csv_log_dir)
        
        csv_logger = CSVLogger(hparam_csv_log_dir, name=None)
        
        synthesizer = Synthesizer(model_name,
                                  feature_dim,
                                  condition_dim,
                                  hparam[0],
                                  int(hparam[0]//2),
                                  hparam[1])
        hparam_df.loc[hparam_no, 'param_count'] = sum(p.numel() for p in synthesizer.model.parameters())
        
        trainer = pl.Trainer(max_epochs=100, # try different values
                             callbacks=[early_stopping,
                                        checkpoint_callback],
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
        
        plot_dir = hparam_csv_log_dir + '/plot'
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        metrics = ['loss', 'mse_loss', 'kl_loss']
        metric_path = hparam_csv_log_dir + '/version_0/metrics.csv'
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
    
        hparam_df.loc[hparam_no, 'val_mse_loss'] = metric_df['val_mse_loss'].min()
        
        print('hparam_df.loc[hparam_no, val_mse_loss]', hparam_df.loc[hparam_no, 'val_mse_loss'])
        print('trainer.checkpoint_callback.best_model_score', trainer.checkpoint_callback.best_model_score)
        assert hparam_df.loc[hparam_no, 'val_mse_loss'] == trainer.checkpoint_callback.best_model_score
        
        
        '''
        predicted = trainer.predict(ckpt_path='best',
                                     dataloaders=[train_loader, val_loader])
        out = []
        samples = []
        for dataloader_idx in range(len(predicted)):
            for batch_idx in range(len(predicted[dataloader_idx])):
                samples.extend(predicted[dataloader_idx][batch_idx][0])
                out.append(predicted[dataloader_idx][batch_idx][1])
                
        out = torch.concat(out, dim=0)
        out = out.cpu().detach().numpy()
        
        out_df = pd.DataFrame(out,
                              columns=train_set.feature_names,
                              index=samples)
        out_df.index.name = 'Sample'
        mic_df = out_df[train_set.microbiome_names]
        met_df = out_df[train_set.metabolome_names]
        mic_df.to_csv('reconstructed_microbiome.tsv', sep='\t', index=True)
        met_df.to_csv('reconstructed_metabolome.tsv', sep='\t', index=True)'''
        
        best_model = Synthesizer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        best_model.eval()
        
        condition, z, out = best_model.sample_data(train_set.cohort_pct, syn_sample_count)
        condition = condition.cpu().detach().numpy()
        samples = ['sample-'+str(i) for i in range(1, condition.shape[0]+1)]
        condition_df = pd.DataFrame(condition,
                                   index=samples,
                                   columns=train_set.cohort_names)
        condition_df.index.name = 'Sample'
        condition_df.to_csv('synthetic_metadata.tsv', sep='\t', index=True)
        
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
        
        mic_df = out_df[train_set.microbiome_names]
        met_df = out_df[train_set.metabolome_names]

        mic_df.to_csv('synthetic_microbiome.tsv', sep='\t', index=True)
        met_df.to_csv('synthetic_metabolome.tsv', sep='\t', index=True)
        
        os.chdir('..')
        print('\n')
    hparam_df.to_csv(csv_log_dir + '/hyperparameters.tsv', sep='\t', index=True)
    best_hparam = hparam_df['val_mse_loss'].idxmin()
    with open(csv_log_dir + '/best_hparam.txt', 'w') as fp:
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
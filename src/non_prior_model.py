import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from itertools import product
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import shutil
import argparse
from pathlib import Path
import os, sys
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbiome_path", type=str,
                        help="path to the micriobiome data in .tsv formnat",
                        required=True, default=None)
    parser.add_argument("--metabolome_path", type=str,
                        help="path to the metabolome data in .tsv formnat",
                        required=True, default=None)
    parser.add_argument("--metadata_path", type=str,
                        help="path to the metadata data in .tsv formnat",
                        required=True, default=None)
    parser.add_argument("-o", "--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

class MicrobiomeMetabolomeDataset(Dataset):
    def __init__(self, microbiome_path, metabolome_path,condition_path, device):
        print('MicrobiomeMetabolomeDataset')
        condition_df = pd.read_csv(condition_path, sep='\t', index_col='Sample')
    
        microbiome_df = pd.read_csv(microbiome_path, sep='\t', index_col='Sample')
        microbiome_df = microbiome_df.reindex(condition_df.index)

        metabolome_df = pd.read_csv(metabolome_path, sep='\t', index_col='Sample')
        metabolome_df = metabolome_df.reindex(condition_df.index)
        
        self.condition = torch.from_numpy(condition_df.to_numpy()).float().to(device)
        self.microbiome = torch.from_numpy(microbiome_df.to_numpy()).float().to(device)
        self.metabolome = torch.from_numpy(metabolome_df.to_numpy()).float().to(device)
        
        self.microbiome_names = list(microbiome_df.columns)
        self.metabolome_names = list(metabolome_df.columns)
        self.sample_names = list(condition_df.index)
        
        self.device = device
        self.length = condition_df.shape[0]
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        condition = self.condition[idx, :]
        microbiome = self.microbiome[idx, :]
        metabolome = self.metabolome[idx, :]

        return {
            'condition': condition,
            'microbiome': microbiome,
            'metabolome': metabolome
        }
    def get_microbiome_names(self):
        return self.microbiome_names

    def get_metabolome_names(self):
        return self.metabolome_names
    
    def get_sample_names(self):
        return self.sample_names
        

class ModuleEncoder(nn.Module):
    def __init__(self, input_dim, condition_dim, output_dim):
        super(ModuleEncoder, self).__init__()
        self.lin_i = nn.Linear(input_dim, output_dim)
        self.lin_c = nn.Linear(condition_dim, output_dim)
        self.lin = nn.Linear(2*output_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, i, c):
        hidden_i = self.relu(self.lin_i(i))
        hidden_c = self.relu(self.lin_c(c))
        hidden_ic = torch.concat([hidden_i, hidden_c], dim=1)
        hidden = self.relu(self.lin(hidden_ic))
        return hidden
    
class CoupledEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CoupledEncoder, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        hidden = self.relu(self.lin(x))
        return hidden

class LatentEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LatentEncoder, self).__init__()
        self.microbiome_mean = nn.Linear(input_dim, output_dim)
        self.microbiome_logvar = nn.Linear(input_dim, output_dim)
        
        self.metabolome_mean = nn.Linear(input_dim, output_dim)
        self.metabolome_logvar = nn.Linear(input_dim, output_dim)
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return z
    
    def forward(self, x):
        microbiome_mean = self.microbiome_mean(x)
        microbiome_logvar = self.microbiome_logvar(x)
        
        metabolome_mean = self.metabolome_mean(x)
        metabolome_logvar = self.metabolome_logvar(x)
        
        microbiome_z = self.reparameterize(microbiome_mean, microbiome_logvar)
        metabolome_z = self.reparameterize(metabolome_mean, metabolome_logvar)
        
        return microbiome_mean, microbiome_logvar, microbiome_z, metabolome_mean, metabolome_logvar, metabolome_z

class LatentDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LatentDecoder, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        hidden = self.lin(x)
        return x
    
class CoupledDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CoupledDecoder, self).__init__()
        self.microbiome_lin = nn.Linear(input_dim, output_dim)
        self.metabolome_lin = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        microbiome_hidden = self.microbiome_lin(x)
        metabolome_hidden = self.metabolome_lin(x)
        return microbiome_hidden, metabolome_hidden   

class ModuleDecoder(nn.Module):
    def __init__(self, input_dim, condition_dim, output_dim):
        super(ModuleDecoder, self).__init__()
        self.lin_i = nn.Linear(input_dim, output_dim)
        self.lin_c = nn.Linear(condition_dim, output_dim)
        self.lin = nn.Linear(2*output_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, i, c):
        hidden_i = self.relu(self.lin_i(i))
        hidden_c = self.relu(self.lin_c(c))
        hidden_ic = torch.concat([hidden_i, hidden_c], dim=1)
        output = self.relu(self.lin(hidden_ic))
        return output
    
class ConditionalCoupledVariationalAutoencoder(nn.Module):
    def __init__(self, microbiome_dim, metabolome_dim, condition_dim, hidden_dim, latent_dim):
        super(ConditionalCoupledVariationalAutoencoder, self).__init__()
        self.microbiome_dim = microbiome_dim
        self.metabolome_dim = metabolome_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.microbiome_encoder = ModuleEncoder(microbiome_dim, condition_dim, hidden_dim)
        self.metabolome_encoder = ModuleEncoder(metabolome_dim, condition_dim, hidden_dim)
        
        self.coupled_encoder = CoupledEncoder(2*hidden_dim, hidden_dim)
        self.latent_encoder = LatentEncoder(hidden_dim, latent_dim)
        
        self.latent_decoder = LatentDecoder(2*latent_dim, hidden_dim)
        self.coupled_decoder = CoupledDecoder(hidden_dim, hidden_dim)
        
        self.microbiome_decoder = ModuleDecoder(hidden_dim, condition_dim, microbiome_dim)
        self.metabolome_decoder = ModuleDecoder(hidden_dim, condition_dim, metabolome_dim)
    
    def encode(self, microbiome_input, metabolome_input, condition_input):
        microbiome_hidden = self.microbiome_encoder(microbiome_input, condition_input)
        metabolome_hidden = self.metabolome_encoder(metabolome_input, condition_input)
        
        modular_hidden = torch.concat([microbiome_hidden, metabolome_hidden], dim=1)
        coupled_hidden = self.coupled_encoder(modular_hidden)
        return self.latent_encoder(coupled_hidden)
        
    def decode(self, microbiome_z, metabolome_z, condition_input):
        coupled_z = torch.concat([microbiome_z, metabolome_z], dim=1)
        coupled_hidden = self.latent_decoder(coupled_z)
        microbiome_hidden, metabolome_hidden = self.coupled_decoder(coupled_hidden)
        microbiome_output = self.microbiome_decoder(microbiome_hidden, condition_input)
        metabolome_output = self.metabolome_decoder(metabolome_hidden, condition_input)
        return microbiome_output, metabolome_output
        
    def forward(self, microbiome_input, metabolome_input, condition_input):
        microbiome_mean, microbiome_logvar, microbiome_z, metabolome_mean, metabolome_logvar, metabolome_z = self.encode(microbiome_input, metabolome_input, condition_input)
        microbiome_output, metabolome_output = self.decode(microbiome_z, metabolome_z, condition_input)
        return microbiome_mean, microbiome_logvar, microbiome_output, metabolome_mean, metabolome_logvar, metabolome_output
    
    def sample(self, sample_count, condition_input):
        microbiome_z = torch.randn(sample_count, self.latent_dim)
        metabolome_z = torch.randn(sample_count, self.latent_dim)
        samples = self.decode(microbiome_z, metabolome_z, condition_input)
        return samples
    
class PlModel(pl.LightningModule):
    def __init__(self, microbiome_dim, metabolome_dim, condition_dim, hidden_dim, latent_dim, lr, microbiome_kl_coeff, metabolome_kl_coeff):
        super(PlModel, self).__init__()
        self.save_hyperparameters()
        self.model = ConditionalCoupledVariationalAutoencoder(microbiome_dim, metabolome_dim, condition_dim, hidden_dim, latent_dim)
        self.lr = lr
        self.microbiome_kl_coeff = microbiome_kl_coeff
        self.metabolome_kl_coeff = metabolome_kl_coeff
    
    def reconstruction_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def kullback_leibler_loss(self, mean, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)

    def forward(self, microbiome_input, metabolome_input, condition_input):
        return self.model(microbiome_input, metabolome_input, condition_input)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        microbiome_mean, microbiome_logvar, microbiome_output, metabolome_mean, metabolome_logvar, metabolome_output = self.forward(batch['microbiome'], batch['metabolome'], batch['condition'])
        
        microbiome_recon_loss = self.reconstruction_loss(microbiome_output, batch['microbiome'])
        microbiome_kl_loss = self.kullback_leibler_loss(microbiome_mean, microbiome_logvar)
        microbiome_loss = microbiome_recon_loss + self.microbiome_kl_coeff * microbiome_kl_loss

        metabolome_recon_loss = self.reconstruction_loss(metabolome_output, batch['metabolome'])
        metabolome_kl_loss = self.kullback_leibler_loss(metabolome_mean, metabolome_logvar)
        metabolome_loss = metabolome_recon_loss + self.metabolome_kl_coeff * metabolome_kl_loss

        loss = microbiome_loss + metabolome_loss
        
        self.log_dict({('train_both_total'): loss.item(),
                      ('train_microbiome_recon'): microbiome_recon_loss.item(),
                      ('train_microbiome_kl'): microbiome_kl_loss.item(),
                      ('train_microbiome_total'): microbiome_loss.item(),
                      ('train_metabolome_recon'): metabolome_recon_loss.item(),
                      ('train_metabolome_kl'): metabolome_kl_loss.item(),
                      ('train_metabolome_total'): metabolome_loss.item()},
                      on_step=False,
                      on_epoch=True)
        
        '''self.log_dict({('train', 'both', 'total'): loss.item(),
                      ('train', 'microbiome', 'recon'): microbiome_recon_loss.item(),
                      ('train', 'microbiome', 'kl'): microbiome_kl_loss.item(),
                      ('train', 'microbiome', 'total'): microbiome_loss.item(),
                      ('train', 'metabolome', 'recon'): metabolome_recon_loss.item(),
                      ('train', 'metabolome', 'kl'): metabolome_kl_loss.item(),
                      ('train', 'metabolome', 'total'): metabolome_loss.item()},
                      on_step=True,
                      on_epoch=True)'''
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        microbiome_mean, microbiome_logvar, microbiome_output, metabolome_mean, metabolome_logvar, metabolome_output = self.forward(batch['microbiome'], batch['metabolome'], batch['condition'])
        
        microbiome_recon_loss = self.reconstruction_loss(microbiome_output, batch['microbiome'])
        microbiome_kl_loss = self.kullback_leibler_loss(microbiome_mean, microbiome_logvar)
        microbiome_loss = microbiome_recon_loss + self.microbiome_kl_coeff * microbiome_kl_loss

        metabolome_recon_loss = self.reconstruction_loss(metabolome_output, batch['metabolome'])
        metabolome_kl_loss = self.kullback_leibler_loss(metabolome_mean, metabolome_logvar)
        metabolome_loss = metabolome_recon_loss + self.metabolome_kl_coeff * metabolome_kl_loss

        loss = microbiome_loss + metabolome_loss
        
        if(dataloader_idx == 0):
            self.log_dict({('val_both_total'): loss.item(),
                          ('val_microbiome_recon'): microbiome_recon_loss.item(),
                          ('val_microbiome_kl'): microbiome_kl_loss.item(),
                          ('val_microbiome_total'): microbiome_loss.item(),
                          ('val_metabolome_recon'): metabolome_recon_loss.item(),
                          ('val_metabolome_kl'): metabolome_kl_loss.item(),
                          ('val_metabolome_total'): metabolome_loss.item()},
                          on_step=False,
                          on_epoch=True)
        elif(dataloader_idx == 1):
            self.log_dict({('test_both_total'): loss.item(),
                      ('test_microbiome_recon'): microbiome_recon_loss.item(),
                      ('test_microbiome_kl'): microbiome_kl_loss.item(),
                      ('test_microbiome_total'): microbiome_loss.item(),
                      ('test_metabolome_recon'): metabolome_recon_loss.item(),
                      ('test_metabolome_kl'): metabolome_kl_loss.item(),
                      ('test_metabolome_total'): metabolome_loss.item()},
                      on_step=False,
                      on_epoch=True)
        
        '''self.log_dict({('val', 'both', 'total'): loss.item(),
                      ('val', 'microbiome', 'recon'): microbiome_recon_loss.item(),
                      ('val', 'microbiome', 'kl'): microbiome_kl_loss.item(),
                      ('val', 'microbiome', 'total'): microbiome_loss.item(),
                      ('val', 'metabolome', 'recon'): metabolome_recon_loss.item(),
                      ('val', 'metabolome', 'kl'): metabolome_kl_loss.item(),
                      ('val', 'metabolome', 'total'): metabolome_loss.item()},
                      on_step=True,
                      on_epoch=True)'''
        return loss
    
    def test_step(self, batch, batch_idx):
        microbiome_mean, microbiome_logvar, microbiome_output, metabolome_mean, metabolome_logvar, metabolome_output = self.forward(batch['microbiome'], batch['metabolome'], batch['condition'])
        
        microbiome_recon_loss = self.reconstruction_loss(microbiome_output, batch['microbiome'])
        microbiome_kl_loss = self.kullback_leibler_loss(microbiome_mean, microbiome_logvar)
        microbiome_loss = microbiome_recon_loss + self.microbiome_kl_coeff * microbiome_kl_loss

        metabolome_recon_loss = self.reconstruction_loss(metabolome_output, batch['metabolome'])
        metabolome_kl_loss = self.kullback_leibler_loss(metabolome_mean, metabolome_logvar)
        metabolome_loss = metabolome_recon_loss + self.metabolome_kl_coeff * metabolome_kl_loss

        loss = microbiome_loss + metabolome_loss
        
        self.log_dict({('test_both_total'): loss.item(),
                      ('test_microbiome_recon'): microbiome_recon_loss.item(),
                      ('test_microbiome_kl'): microbiome_kl_loss.item(),
                      ('test_microbiome_total'): microbiome_loss.item(),
                      ('test_metabolome_recon'): metabolome_recon_loss.item(),
                      ('test_metabolome_kl'): metabolome_kl_loss.item(),
                      ('test_metabolome_total'): metabolome_loss.item()},
                      on_step=False,
                      on_epoch=True)
        
        '''self.log_dict({('test', 'both', 'total'): loss.item(),
                      ('test', 'microbiome', 'recon'): microbiome_recon_loss.item(),
                      ('test', 'microbiome', 'kl'): microbiome_kl_loss.item(),
                      ('test', 'microbiome', 'total'): microbiome_loss.item(),
                      ('test', 'metabolome', 'recon'): metabolome_recon_loss.item(),
                      ('test', 'metabolome', 'kl'): metabolome_kl_loss.item(),
                      ('test', 'metabolome', 'total'): metabolome_loss.item()},
                      on_step=True,
                      on_epoch=True)'''
        return loss
    
    def predict_step(self, batch, batch_idx):
        microbiome_mean, microbiome_logvar, microbiome_output, metabolome_mean, metabolome_logvar, metabolome_output = self.forward(batch['microbiome'], batch['metabolome'], batch['condition'])
        
        '''microbiome_recon_loss = self.reconstruction_loss(microbiome_output, batch['microbiome'])
        microbiome_kl_loss = self.kullback_leibler_loss(microbiome_mean, microbiome_logvar)
        microbiome_loss = microbiome_recon_loss + self.microbiome_kl_coeff * microbiome_kl_loss

        metabolome_recon_loss = self.reconstruction_loss(metabolome_output, batch['metabolome'])
        metabolome_kl_loss = self.kullback_leibler_loss(metabolome_mean, metabolome_logvar)
        metabolome_loss = metabolome_recon_loss + self.metabolome_kl_coeff * metabolome_kl_loss

        loss = microbiome_loss + metabolome_loss'''
        
        '''self.log_dict({('predict_both_total'): loss.item(),
                      ('predict_microbiome_recon'): microbiome_recon_loss.item(),
                      ('predict_microbiome_kl'): microbiome_kl_loss.item(),
                      ('predict_microbiome_total'): microbiome_loss.item(),
                      ('predict_metabolome_recon'): metabolome_recon_loss.item(),
                      ('predict_metabolome_kl'): metabolome_kl_loss.item(),
                      ('predict_metabolome_total'): metabolome_loss.item()},
                      on_step=True,
                      on_epoch=True)'''
        
        '''self.log_dict({('predict', 'both', 'total'): loss.item(),
                      ('predict', 'microbiome', 'recon'): microbiome_recon_loss.item(),
                      ('predict', 'microbiome', 'kl'): microbiome_kl_loss.item(),
                      ('predict', 'microbiome', 'total'): microbiome_loss.item(),
                      ('predict', 'metabolome', 'recon'): metabolome_recon_loss.item(),
                      ('predict', 'metabolome', 'kl'): metabolome_kl_loss.item(),
                      ('predict', 'metabolome', 'total'): metabolome_loss.item()},
                      on_step=True,
                      on_epoch=True)'''
        
        return microbiome_mean, microbiome_logvar, microbiome_output, metabolome_mean, metabolome_logvar, metabolome_output
    
    def sample(self, sample_count, condition_input, microbiome_path, metabolome_path, microbiome_names, metabolome_names):
        microbiome_sample, metabolome_sample =  self.model.sample(sample_count)
        microbiome_df = pd.DataFrame(microbiome_sample.cpu().detach().to_numpy(),
                                     columns=microbiome_names)
        microbiome_df.to_csv(microbiome_path, sep='\t', index=False)
        
        metabolome_df = pd.DataFrame(metabolome_sample.cpu().detach().to_numpy(),
                                     columns=metabolome_names)
        metabolome_df.to_csv(metabolome_path, sep='\t', index=False)
        
def plot_metric(x, train_y, val_y, test_y, xlabel, ylabel, title, png_path):
    plt.plot(x, train_y, label='Train')
    plt.plot(x, val_y, label='Val')
    plt.plot(x, test_y, label='Test')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(png_path)
    plt.close()
        
def generate_synthetic_data(microbiome_path, metabolome_path, condition_path):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    dataset = MicrobiomeMetabolomeDataset(microbiome_path, metabolome_path, condition_path, device)
    
    train_pct = 0.7
    val_pct = 0.15
    test_pct = 0.15
    
    train_set, val_set, test_set =  random_split(dataset, [train_pct, val_pct, test_pct])
    print('dataset', dataset.__len__(),
          'train_set', train_set.__len__(),
          'val_set', val_set.__len__(),
          'test_set', test_set.__len__())
    
    batch_size = 32
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    data = dataset[0]
    
    microbiome_dim = data['microbiome'].shape[0]
    metabolome_dim = data['metabolome'].shape[0]
    condition_dim = data['condition'].shape[0]
    
    print('microbiome_dim', microbiome_dim,
         'metabolome_dim', metabolome_dim,
         'condition_dim', condition_dim)
    
    #hidden_dim = [8, 16]
    #lr = [1e-1, 1e-3]
    #kl_coeff = (0.5, 2)
    
    hidden_dim = [4, 8, 16]
    lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    kl_coeff = [0.5, 1, 2]
    label = ['haparm-'+str(i) for i in range(len(hidden_dim)*len(lr)*len(kl_coeff)*len(kl_coeff))]
    
    hparams = list(product(label, hidden_dim, lr, kl_coeff, kl_coeff))
    
    log_dir = os.getcwd() + '/logs'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    hparam_df = pd.DataFrame(hparams, columns=['hparam_no', 'hdim', 'lr', 'coeff_mic', 'coeff_met'])
    hparam_df.to_csv(log_dir + '/hyperparameters.tsv', sep='\t', index=False)
    #index = []
    #data = []
    
    for hparam in hparams:
        #index.append(hparam)
        Path(hparam[0]).mkdir(parents=True, exist_ok=True)
        os.chdir(hparam[0])
        
        early_stopping = EarlyStopping(monitor='val_both_total/dataloader_idx_0', mode='min', patience=3)
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_both_total/dataloader_idx_0", mode="min")
        
        hparam_log_dir = log_dir + '/' + hparam[0]
        if(os.path.exists(hparam_log_dir)):
            shutil.rmtree(hparam_log_dir)
        
        logger = CSVLogger(hparam_log_dir, name=None)
        
        plModel = PlModel(microbiome_dim, metabolome_dim, condition_dim, hparam[1], hparam[1]//2, hparam[2], hparam[3], hparam[4])
        trainer = pl.Trainer(max_epochs=250,
                             callbacks=[early_stopping, checkpoint_callback],
                             log_every_n_steps=1,
                             accelerator="gpu",
                             devices=1,
                             enable_checkpointing=True,
                             logger=logger,
                             check_val_every_n_epoch=1,
                             val_check_interval=1.0,
                             enable_progress_bar=False)
        
        trainer.fit(plModel, train_dataloaders=train_loader, val_dataloaders=[val_loader, test_loader])
        trainer.test(ckpt_path='best', dataloaders=test_loader)
        prediction = trainer.predict(ckpt_path='best',
                                     dataloaders=data_loader)
        
        predicted_values = ['microbiome_mean', 'microbiome_logvar', 'microbiome_output', 
                            'metabolome_mean', 'metabolome_logvar', 'metabolome_output']
        
        sample_names = dataset.get_sample_names()
        
        pred_dir = 'prediction'
        Path(pred_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(pred_dir)
        
        for i in range(len(predicted_values)):
            batches = []
            for batch in range(len(prediction)):
                batches.append(prediction[batch][i])
            pred = torch.concat(batches, dim=0)
            pred = pred.cpu().detach().numpy()
            
            if(predicted_values[i] == 'microbiome_output'):
                columns = dataset.get_microbiome_names()
            elif(predicted_values[i] == 'metabolome_output'):
                columns = dataset.get_metabolome_names()
            else:
                columns = None
            pred_df = pd.DataFrame(pred, columns=columns, index=sample_names)
            pred_df.index.name = 'Sample'
            pred_df.to_csv(predicted_values[i]+'.tsv', sep='\t', index=True)
            
        os.chdir('..')
        
        plot_dir = 'plot'
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(plot_dir)
        
        metrics = ['both_total', 'microbiome_total', 'metabolome_total',
                   'microbiome_kl', 'microbiome_recon', 'metabolome_kl', 'metabolome_recon']
        
        metric_df = pd.read_csv(hparam_log_dir + '/version_0/metrics.csv',
                                sep=',', index_col='epoch')
        
        last_epoch = metric_df.index.max()
        
        epochs = list(range(last_epoch))
        
        for metric in metrics:
            train_metric = 'train_' + metric
            val_metric = 'val_' + metric + '/dataloader_idx_0'
            test_metric = 'test_' + metric + '/dataloader_idx_1'
            
            train_y = list(metric_df[train_metric].dropna().loc[epochs])
            val_y = list(metric_df[val_metric].dropna().loc[epochs])
            test_y = list(metric_df[test_metric].dropna().loc[epochs])
            
            plot_metric(epochs, train_y, val_y, test_y, 'Epoch', metric, metric + ' across epochs', metric + '.png')
            
        os.chdir('../..')
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/non_prior_model.log', 'w')
    
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    original_stderr = sys.stderr
    sys.stderr = log_file
    
    print(args)
    
    generate_synthetic_data(args.microbiome_path, args.metabolome_path, args.metadata_path)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    log_file.flush()
    log_file.close()
    
if __name__ == "__main__":
    main(parse_args())

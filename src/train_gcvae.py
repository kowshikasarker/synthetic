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

from model import GCVAE
from loss import KLLoss
from annealer import Annealer, AnnealerStep

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop

from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import ToUndirected

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

def parse_args():
    parser = argparse.ArgumentParser()
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

class GCVAEDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)
        
    @property
    def raw_file_names(self):
        return ['feature.tsv', 'condition.tsv', 'edge.tsv']
    
    @property
    def processed_file_names(self):
        condition_df = pd.read_csv(self.raw_dir + '/condition.tsv', sep='\t', index_col='Sample')
        return [sample + '.pt' for sample in condition_df.index]

    def process(self):
        feature_df = pd.read_csv(self.raw_dir + '/feature.tsv', sep='\t', index_col='Sample')
        condition_df = pd.read_csv(self.raw_dir + '/condition.tsv', sep='\t', index_col='Sample')
        cohort_pct = np.sum(condition_df.to_numpy(), axis=0)
        cohort_pct = cohort_pct / np.sum(cohort_pct)
        
        edge_df = pd.read_csv(self.raw_dir + '/edge.tsv', sep='\t')
        nodes = list(feature_df.columns)
        num_nodes = len(nodes)
        nid = list(range(len(nodes)))
        node_id = dict(zip(nodes, nid))
        with open(self.processed_dir + '/node_id.json', 'w') as fp:
            json.dump(node_id, fp)
            
        feature_df = feature_df.rename(columns=node_id)
        feature_df = feature_df[nid]
        
        edge_df = edge_df[(edge_df.Node1.isin(nodes)) & (edge_df.Node2.isin(nodes))]
        
        edge_df['Node1_ID'] = edge_df['Node1'].map(node_id)
        edge_df['Node2_ID'] = edge_df['Node2'].map(node_id)
        
        edge_index = edge_df[['Node1_ID', 'Node2_ID']].to_numpy().T
        
        self.samples = list(condition_df.index)
        self.length = len(self.samples)
        self.cohort_pct = cohort_pct
        self.feature_names = nodes
        self.cohort_names = list(condition_df.columns)
        self.graph_edges = torch.from_numpy(edge_index).long()
        
        for sample in self.samples:
            data = Data()
            data.num_nodes = num_nodes
            feature = feature_df.loc[sample, :].to_numpy()
            feature = feature.reshape((-1, 1))
            data.feature = torch.from_numpy(feature).float()
            
            condition = condition_df.loc[sample, :].to_numpy()
            condition = np.tile(condition, (feature.shape[0], 1))
            data.condition = torch.from_numpy(condition).float()
            
            data.edge_index = torch.from_numpy(edge_index).long()
            data.sample = sample
            data = ToUndirected()(data)
            torch.save(data, self.processed_dir + '/' + sample + '.pt')
            
    def len(self):
        return len(self.samples)

    def get(self, idx):
        return torch.load(self.processed_dir + '/' + self.samples[idx] + '.pt')
    
def create_pyg_dataset(dataset_dir,
                       feature_path,
                       condition_path,
                       edge_path):
    raw_dir = dataset_dir + '/raw'
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    
    processed_dir = dataset_dir + '/processed'
    if(os.path.exists(processed_dir)):
        rmtree(processed_dir, ignore_errors=True)
    
    copyfile(feature_path, raw_dir + '/feature.tsv')
    copyfile(condition_path, raw_dir + '/condition.tsv')
    copyfile(edge_path, raw_dir + '/edge.tsv')
        
    dataset = GCVAEDataset(dataset_dir)
    
    return dataset

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
            'combined_hidden': GCVAE,
        }
        loss_map = {
            'kl': KLLoss,
            'mse': torch.nn.MSELoss,
        }
        self.model = model_map[model_name](feature_dim,
                                           condition_dim,
                                           hidden_dim,
                                           latent_dim)
        print('self.model', self.model)
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.kl_module = KLLoss()
        self.mse_module = torch.nn.MSELoss()
        self.annealer = Annealer(total_steps=10, cyclical=True, shape='linear')
        
    def sample_data_batch(self, cohort_pct, sample_count, graph_edges, feature_count):
        assert graph_edges.max() == feature_count - 1
        
        edge_index = []
        for i in range(sample_count):
            edge_index.append(torch.add(graph_edges, feature_count))
        edge_index = torch.cat(edge_index, dim=1)
        print('graph_edges', graph_edges.shape,
              'sample_count', sample_count,
              'edge_index', edge_index.shape)
        
        cohort_count = np.round(np.multiply(cohort_pct, sample_count)).astype(int)
        cohort = np.zeros(shape=(np.sum(cohort_count), cohort_count.size))
        past_samples = 0
        for cohort_i in range(cohort_count.size):
            cohort[past_samples: past_samples+cohort_count[cohort_i], cohort_i] = 1
            past_samples += cohort_count[cohort_i]
        cohort = np.repeat(cohort, feature_count, axis=0)
        condition = torch.from_numpy(cohort).float()
        condition = condition.to(self.device)
        edge_index = edge_index.to(self.device).long()
        z, samples = self.model.sample(condition, edge_index)
        return condition, z, samples
    
    def sample_data(self, cohort_pct, sample_count, graph_edges, feature_count):
        batch_max_size = 64
        batch_condition = []
        batch_z = []
        batch_samples = []
        
        batch_sizes = [batch_max_size] * (sample_count // batch_max_size) + [sample_count % batch_max_size]
        
        for batch_size in batch_sizes:
            condition, z, samples = self.sample_data_batch(cohort_pct, batch_size, graph_edges, feature_count)
            batch_condition.append(condition)
            batch_z.append(z)
            batch_samples.append(samples)
            
        
        condition = torch.cat(batch_condition, 0)
        z = torch.cat(batch_z, 0)
        samples = torch.cat(batch_samples, 0)
        
        return condition, z, samples
    
    def forward(self, feature, condition, edge_index):
        feature = feature.to(self.device)
        condition = condition.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.model(feature, condition, edge_index)
    
    def configure_optimizers(self):
        return Adam(self.parameters(),
                    lr=self.lr)
    
    def calc_loss(self,
                  pred_feat,
                  true_feat,
                  mean,
                  logvar,
                  prefix):
        loss = 0
        loss_dict = {}
        
        kl_loss = self.kl_module(mean, logvar)
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
        z, mean, logvar, out = self.forward(batch.feature, batch.condition, batch.edge_index)
        loss = self.calc_loss(out, batch.feature, mean, logvar, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        z, mean, logvar, out = self.forward(batch.feature, batch.condition, batch.edge_index)
        loss = self.calc_loss(out, batch.feature, mean, logvar, 'val')
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
                         names=['memory.used',
                                'memory.free'],
                         skiprows=1)
    gpu_df.to_csv('gpu_memory.tsv', sep='\t')
    gpu_df['memory.used'] = gpu_df['memory.used'].str.replace(" MiB", "").astype(int)
    gpu_df['memory.free'] = gpu_df['memory.free'].str.replace(" MiB", "").astype(int)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_id = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(gpu_id, gpu_df.iloc[gpu_id]['memory.free']))
    return gpu_id  

def generate_synthetic_data(**kwargs):
    train_set = create_pyg_dataset(os.getcwd() + '/train_dataset',
                                   kwargs['train_feature_path'],
                                   kwargs['train_condition_path'],
                                   kwargs['edge_path'])
    val_set = create_pyg_dataset(os.getcwd() + '/val_dataset',
                                 kwargs['val_feature_path'],
                                 kwargs['val_condition_path'],
                                 kwargs['edge_path'])
    
    print('train_set', train_set.len(),
          'val_set', val_set.len())
    
    data = train_set[0]
    feature_dim = data.feature.shape[1]
    condition_dim = data.condition.shape[1]
    
    print('feature_dim', feature_dim, data.feature.shape)
    print('condition_dim', condition_dim, data.condition.shape)
    
    # good result for hidden_dim = 4, lr = 1e-5, batch_size = 256 (both mse_loss and kl_loss decreased)
    hidden_dim = [4, 8]
    lr = [1e-4, 1e-5, 1e-6]
    batch_size = [128, 256]
    
    hparams = list(product(hidden_dim, lr, batch_size))
    hparam_label = ['hparam-'+str(i) for i in range(len(hparams))]
    
    log_dir = os.getcwd() + '/logs'
    if(os.path.exists(log_dir)):
        rmtree(log_dir)
    Path(log_dir).mkdir(parents=True)
    
    hparam_df = pd.DataFrame(hparams,
                             columns=['hidden_dim', 'lr', 'batch_size'],
                             index=hparam_label)
    hparam_df.to_csv(log_dir + '/hyperparameters.tsv', sep='\t', index=True)
    
    
    for i in range(len(hparam_label)):
        hparam_no = hparam_label[i]
        hparam = hparams[i]
        print('\n\n')
        print(hparam_no, hparam, end='\n')
        
        if(os.path.exists(hparam_no)):
            rmtree(hparam_no)
            
        Path(hparam_no).mkdir(parents=True)
        os.chdir(hparam_no)
        
        train_loader = DataLoader(train_set,
                                  batch_size=hparam[2],
                                  shuffle=False) # check shuffle in prediction
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
        
        csv_logger = CSVLogger(hparam_log_dir, name=None)
        
        synthesizer = Synthesizer(kwargs['model_name'],
                                  feature_dim,
                                  condition_dim,
                                  hparam[0],
                                  int(hparam[0] // 2),
                                  hparam[1])
        hparam_df.loc[hparam_no, 'param_count'] = sum(p.numel() for p in synthesizer.model.parameters())
        
        trainer = pl.Trainer(max_epochs=100, # try different values
                             callbacks=[early_stopping,
                                        checkpoint_callback,
                                        annealer_step],
                             log_every_n_steps=1,
                             accelerator="cuda",
                             devices=[get_free_gpu()], # pick the gpu with the max free space -> processes will get distributed across gpus
                             enable_checkpointing=True,
                             logger=[csv_logger],
                             check_val_every_n_epoch=1,
                             val_check_interval=1.0,
                             enable_progress_bar=False,
                             detect_anomaly=True,
                             num_sanity_val_steps=0)
        
        trainer.fit(synthesizer,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
        
        metric_path = hparam_log_dir + '/version_0/metrics.csv'
        metric_df = pd.read_csv(metric_path, sep=',')
        
        hparam_df.loc[hparam_no, 'last_epoch'] = metric_df['epoch'].max()
        
        plot_dir = hparam_log_dir + '/plot'
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        metrics = ['loss', 'kl_loss', 'mse_loss']
        for metric in metrics:
            print(metric)
            train_metric = 'train_' + metric
            val_metric = 'val_' + metric
            
            train = metric_df.dropna(subset=train_metric).sort_values(by='epoch')
            val = metric_df.dropna(subset=val_metric).sort_values(by='epoch')
            
            train_x = list(train['epoch'])
            train_y = list(train[train_metric])
            
            val_x = list(val['epoch'])
            val_y = list(val[val_metric])
            
            plot_metric(train_x, val_x, train_y, val_y, 'Epoch', metric, metric + ' across epochs', plot_dir + '/' + metric + '.png')
    
        hparam_df.loc[hparam_no, 'val_mse_loss'] = metric_df['val_mse_loss'].min(skipna=True)
        feature_count = len(train_set.feature_names)
        
        best_model = Synthesizer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        best_model.eval()
        
        condition, z, out = best_model.sample_data(train_set.cohort_pct, kwargs['syn_sample_count'], train_set.graph_edges, feature_count)
        condition = condition.cpu().detach().numpy()
        condition = condition[::feature_count]
        samples = ['sample-'+str(i) for i in range(1, condition.shape[0]+1)]
        condition_df = pd.DataFrame(condition,
                                   index=samples,
                                   columns=train_set.cohort_names)
        condition_df.index.name = 'Sample'
        condition_df.to_csv('synthetic_condition.tsv', sep='\t', index=True)
        
        writer = pd.ExcelWriter('z.xlsx')
        z = z.cpu().detach().numpy()
        for i in range(z.shape[1]):
            zi = z[:, i]
            zi = zi.reshape(-1, feature_count)
            zdf = pd.DataFrame(zi, index=samples, columns=train_set.feature_names)
            zdf.index.name = 'Sample'
            zdf.to_excel(writer, sheet_name='z'+str(i))
        writer.close()
        
        out = out.cpu().detach().numpy()
        out = out.reshape(-1, feature_count)
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
    
    log_file = open(args.out_dir + '/train_gcvae.log', 'w')
    
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
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

from model import CombinedHiddenGCVAE, SeparateHiddenGCVAE
from loss import KLLoss, BCELoss, MultiLossEarlyStopping

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD, RMSprop

from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import ToUndirected

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

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
    
    parser.add_argument("--edge_path", type=str,
                        help="path to the edges in .tsv formnat",
                        required=True, default=None)
    parser.add_argument("--model_name", type=str,
                        choices=['separate_hidden', 'combined_hidden'],
                        help="name of the synthetic data generation model",
                        required=True, default=None)
    parser.add_argument("--syn_sample_count", type=int,
                        help="count of synthetic samples to generate",
                        required=True, default=None)
    
    parser.add_argument("--optim_loss", type=str, nargs="+",
                        choices=['kl', 'mse'],
                        help="Losses to use for optimizing model parameters",
                        required=True, default=None)
    parser.add_argument("--hparam_loss", type=str,
                        choices=['kl', 'mse'],
                        help="loss to use for tuning hyperparameters",
                        required=True, default=None)
    parser.add_argument("--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

class MicrobiomeMetabolomeGCVAEDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)
        
    @property
    def raw_file_names(self):
        return ['microbiome.tsv', 'metabolome.tsv', 'condition.tsv', 'edge.tsv']
    
    @property
    def processed_file_names(self):
        condition_df = pd.read_csv(self.raw_dir + '/condition.tsv', sep='\t', index_col='Sample')
        return [sample + '.pt' for sample in condition_df.index]

    def process(self):
        microbiome_df = pd.read_csv(self.raw_dir + '/microbiome.tsv', sep='\t', index_col='Sample')
        metabolome_df = pd.read_csv(self.raw_dir + '/metabolome.tsv', sep='\t', index_col='Sample')
        feature_df = pd.concat([microbiome_df, metabolome_df], axis=1) 
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
        self.microbiome_names = list(microbiome_df.columns)
        self.metabolome_names = list(metabolome_df.columns)
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
                       microbiome_path,
                       metabolome_path,
                       metadata_path,
                       edge_path):
    raw_dir = dataset_dir + '/raw'
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    
    processed_dir = dataset_dir + '/processed'
    if(os.path.exists(processed_dir)):
        rmtree(processed_dir, ignore_errors=True)
    
    copyfile(microbiome_path, raw_dir + '/microbiome.tsv')
    copyfile(metabolome_path, raw_dir + '/metabolome.tsv')
    copyfile(metadata_path, raw_dir + '/condition.tsv')
    copyfile(edge_path, raw_dir + '/edge.tsv')
        
    dataset = MicrobiomeMetabolomeGCVAEDataset(dataset_dir)
    
    return dataset

class Synthesizer(pl.LightningModule):
    def __init__(self,
                 model_name,
                 feature_dim,
                 condition_dim,
                 hidden_dim,
                 latent_dim,
                 optim_loss,
                 lr):
        super(Synthesizer, self).__init__()
        self.save_hyperparameters()
        model_map = {
            'separate_hidden': SeparateHiddenGCVAE,
            'combined_hidden': CombinedHiddenGCVAE,
        }
        loss_map = {
            'kl': KLLoss,
            'mse': torch.nn.MSELoss,
            'bce': BCELoss
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
        self.optim_loss = optim_loss
        self.lr = lr
        self.loss_modules = {loss: loss_map[loss]() for loss in optim_loss}
        print('self.loss_modules', self.loss_modules)
        
    def sample_data(self, cohort_pct, sample_count, graph_edges, feature_count):
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
    
    def forward(self, feature, condition, edge_index):
        feature = feature.to(self.device)
        condition = condition.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.model(feature, condition, edge_index)
    
    def configure_optimizers(self):
        return RMSprop(self.parameters(),
                        lr=self.lr,
                        momentum=0.9,
                        weight_decay=0.01)
    
    def calc_loss(self,
                  pred_feat,
                  true_feat,
                  mean,
                  logvar,
                  z,
                  edge_index,
                  prefix):
        loss_dict = {}
        loss = 0
        if ('kl' in self.optim_loss):
            kl_loss = self.loss_modules['kl'](mean, logvar)
            loss = kl_loss/kl_loss.detach() + loss
            loss_dict[prefix + '_kl_loss'] = kl_loss.item()
        if ('mse' in self.optim_loss):
            mse_loss = self.loss_modules['mse'](pred_feat, true_feat)
            loss = mse_loss/mse_loss.detach() + loss
            loss_dict[prefix + '_mse_loss'] = mse_loss.item()
            
        loss_dict[prefix + '_loss'] = loss.item()
        
        print('loss_dict', loss_dict)
        self.log_dict(loss_dict, on_step=False, on_epoch=True)
        return loss
            
    
    def training_step(self, batch, batch_idx):
        print('training_step')
        z, mean, logvar, out = self.forward(batch.feature, batch.condition, batch.edge_index)
        loss = self.calc_loss(out, batch.feature, mean, logvar, z, batch.edge_index, 'train')
            
        for name, param in self.model.named_parameters():
            print (name, torch.isnan(param.data).sum(), torch.isinf(param.data).sum())
            print (name, param.data)
        return loss
    
    def validation_step(self, batch, batch_idx):
        print('validation_step')
        z, mean, logvar, out = self.forward(batch.feature, batch.condition, batch.edge_index)
        loss = self.calc_loss(out, batch.feature, mean, logvar, z, batch.edge_index, 'val')
        for name, param in self.model.named_parameters():
            print (name, torch.isnan(param.data).sum(), torch.isinf(param.data).sum())
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
    gpu_stats = check_output(["nvidia-smi",
                                         "--format=csv",
                                         "--query-gpu=memory.used,memory.free"])
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

def generate_synthetic_data(train_mic_path,
                            train_met_path,
                            train_meta_path,
                            val_mic_path,
                            val_met_path,
                            val_meta_path,
                            edge_path,
                            model_name,
                            syn_sample_count,
                            optim_loss,
                            hparam_loss):
    
    train_set = create_pyg_dataset(os.getcwd() + '/train_dataset',
                                   train_mic_path,
                                   train_met_path,
                                   train_meta_path,
                                   edge_path)
    val_set = create_pyg_dataset(os.getcwd() + '/val_dataset',
                                 val_mic_path,
                                 val_met_path,
                                 val_meta_path,
                                 edge_path)
    
    print('train_set', train_set.len(),
          'val_set', val_set.len())
    
    data = train_set[0]
    feature_dim = data.feature.shape[1]
    condition_dim = data.condition.shape[1]
    
    print('feature_dim', feature_dim, data.feature.shape)
    print('condition_dim', condition_dim, data.condition.shape)
    
    #hidden_dim = [4, 6]
    #lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    #batch_size = [4, 8]
    hidden_dim = [2]
    lr = [1e-6, 1e-7, 1e-8]
    batch_size = [256]
    
    hparams = list(product(hidden_dim, lr, batch_size))
    hparam_label = ['hparam-'+str(i) for i in range(len(hparams))]
    
    csv_log_dir = os.getcwd() + '/logs/csv_logs'
    Path(csv_log_dir).mkdir(parents=True, exist_ok=True)
    
    hparam_df = pd.DataFrame(hparams,
                             columns=['hidden_dim', 'lr', 'batch_size'],
                             index=hparam_label)
    hparam_df.to_csv(csv_log_dir + '/hyperparameters.tsv', sep='\t', index=True)
    
    monitor = ['val_' + loss + '_loss' for loss in optim_loss]
    hparam_loss = 'val_' + hparam_loss + '_loss'
    
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
                                  shuffle=True) # check shuffle in prediction
        val_loader = DataLoader(val_set,
                                batch_size=hparam[2],
                                shuffle=False)
        
        early_stopping = MultiLossEarlyStopping(monitor=monitor,
                                                mode=['min']*len(monitor),
                                                patience=[5]*len(monitor),
                                                min_delta=[1e-5]*len(monitor),
                                                check_finite=[True]*len(monitor))
        
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              monitor=hparam_loss,
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
                                  optim_loss,
                                  hparam[1])
        hparam_df.loc[hparam_no, 'param_count'] = sum(p.numel() for p in synthesizer.model.parameters())
        
        trainer = pl.Trainer(max_epochs=-1, # try different values
                             callbacks=[early_stopping,
                                        checkpoint_callback],
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
        
        metric_path = hparam_csv_log_dir + '/version_0/metrics.csv'
        metric_df = pd.read_csv(metric_path, sep=',')
        
        hparam_df.loc[hparam_no, 'last_epoch'] = metric_df['epoch'].max()
        
        plot_dir = hparam_csv_log_dir + '/plot'
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        metrics = ['loss'] + [loss_name + '_loss' for loss_name in optim_loss]
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
    
        hparam_df.loc[hparam_no, hparam_loss] = metric_df[hparam_loss].min(skipna=True)
        feature_count = len(train_set.feature_names)
        
        best_model = Synthesizer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        best_model.eval()
        
        condition, z, out = best_model.sample_data(train_set.cohort_pct, syn_sample_count, train_set.graph_edges, feature_count)
        condition = condition.cpu().detach().numpy()
        condition = condition[::feature_count]
        samples = ['sample-'+str(i) for i in range(1, condition.shape[0]+1)]
        condition_df = pd.DataFrame(condition,
                                   index=samples,
                                   columns=train_set.cohort_names)
        condition_df.index.name = 'Sample'
        
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
        mic_df = out_df[train_set.microbiome_names]
        met_df = out_df[train_set.metabolome_names]
        condition_df.to_csv('synthetic_metadata.tsv', sep='\t', index=True)
        mic_df.to_csv('synthetic_microbiome.tsv', sep='\t', index=True)
        met_df.to_csv('synthetic_metabolome.tsv', sep='\t', index=True)
        
        os.chdir('..')
        print('\n')
    hparam_df.to_csv(csv_log_dir + '/hyperparameters.tsv', sep='\t', index=True)
    best_hparam = hparam_df[hparam_loss].idxmin()
    with open(csv_log_dir + '/best_hparam.txt', 'w') as fp:
        fp.write(best_hparam)
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/train_GCVAE.log', 'w')
    
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

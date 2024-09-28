import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from typing_extensions import override

class KLLoss(torch.nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    
    def forward(self, mean, logvar):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim = 1), dim = 0)
        print('mean', mean.shape, 'logvar', logvar.shape, 'kl_loss', kl_loss.shape)
        return kl_loss
    
class BCELoss(torch.nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        
    def get_adjacency_matrix(self, num_nodes, edge_index):
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0, :], edge_index[1, :]] = 1
        # check if both direction edges are already included in the graphs
        return adj
    
    def get_edge_weight(self, adj):
        edge_weight = adj.detach().clone()
        edge_weight[adj == 0] = 1 / (adj.numel() - adj.sum())
        edge_weight[adj == 1] = 1 / adj.sum()
        return edge_weight
    
    def forward(self, z, edge_index):
        adj_true = self.get_adjacency_matrix(z.shape[0], edge_index)
        adj_pred = torch.matmul(z, z.t())
        edge_weight = self.get_edge_weight(adj_true)
        bce_loss = F.binary_cross_entropy_with_logits(adj_pred, adj_true, weight=edge_weight)
        print('z', z.shape, 'edge_index', edge_index.shape, 'bce_loss', bce_loss.shape)
        return bce_loss
    
class MultiLossEarlyStopping(Callback):
    def __init__(self, monitor, min_delta, patience, mode):
        print('monitor', monitor)
        print('min_delta', min_delta)
        print('patience', patience)
        print('mode', mode)
        super().__init__()
        self.mode_dict = {"min": torch.lt, "max": torch.gt}
        self.order_dict = {"min": "<", "max": ">"}
        self.monitor = monitor
        
        for m in mode:
            if m not in self.mode_dict:
                raise MisconfigurationException(f"`mode` can be {','.join(self.mode_dict.keys())}, got {m}")
                
        self.min_delta = dict(zip(monitor, min_delta))
        self.patience = dict(zip(monitor, patience))
        self.mode = dict(zip(monitor, mode))
        
        self.best_score = {}
        self.wait_count = {}
        
        torch_inf = torch.tensor(torch.inf)
        
        for m in self.monitor:
            monitor_op = self.mode_dict[self.mode[m]]
            self.min_delta[m] *= 1 if monitor_op == torch.gt else -1
            self.best_score[m] = torch_inf if monitor_op == torch.lt else -torch_inf
            self.wait_count[m] = 0
            
    def validate_condition_metric(self, logs):
        for m in self.monitor:
            monitor_val = logs.get(m)
            if (monitor_val is None):
                raise RuntimeError("Early stopping metric " + m + " absent in logs.")
                return False
            return True
    @override
    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        should_stop_count = 0
        for m in self.monitor:
            current = logs[m].squeeze()
            monitor_op = self.mode_dict[self.mode[m]]
            if monitor_op(current - self.min_delta[m], self.best_score[m].to(current.device)):
                self.best_score[m] = current
                self.wait_count[m] = 0
            else:
                self.wait_count[m] += 1
                if self.wait_count[m] >= self.patience[m]:
                    print("Should stop for", m)
                    should_stop_count += 1
        should_stop = should_stop_count == len(self.monitor)
        
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
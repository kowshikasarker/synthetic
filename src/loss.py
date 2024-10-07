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
        super(BCELoss, self).__init__()
        
    def get_adjacency_matrix(self, num_nodes, edge_index):
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0, :], edge_index[1, :]] = 1
        adj = adj.to(edge_index.device)
        # check if both direction edges are already included in the graphs
        return adj
    
    def get_edge_weight(self, adj):
        edge_weight = adj.detach().clone()
        edge_weight[adj == 0] = 1 / (adj.numel() - adj.sum())
        edge_weight[adj == 1] = 1 / adj.sum()
        edge_weight = edge_weight.to(adj.device)
        return edge_weight
    
    def forward(self, z, edge_index):
        adj_true = self.get_adjacency_matrix(z.shape[0], edge_index)
        adj_pred = torch.matmul(z, z.t())
        edge_weight = self.get_edge_weight(adj_true)
        bce_loss = F.binary_cross_entropy_with_logits(adj_pred, adj_true, weight=edge_weight)
        print('z', z.shape, 'edge_index', edge_index.shape, 'bce_loss', bce_loss.shape)
        return bce_loss
    
'''class MultiLossEarlyStopping(Callback):
    def __init__(self, monitor, min_delta, patience, mode):
        super().__init__()
        self.mode_dict = {"min": torch.lt, "max": torch.gt}
        self.monitor = monitor
                
        self.min_delta = dict(zip(monitor, min_delta))
        self.patience = dict(zip(monitor, patience))
        self.mode = dict(zip(monitor, mode))
        #self.check_finite = dict(zip(monitor, check_finite))
        
        self.best_score = {}
        self.wait_count = {}
        
        torch_inf = torch.tensor(torch.inf)
        
        for m in self.monitor:
            monitor_op = self.mode_dict[self.mode[m]]
            self.min_delta[m] *= 1 if monitor_op == torch.gt else -1
            self.best_score[m] = torch_inf if monitor_op == torch.lt else -torch_inf
            self.wait_count[m] = 0
    
    @override
    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        print('logs', logs, end='\n')
        
        should_stop_count = 0
        for m in self.monitor:
            current = logs[m].squeeze()
            if(self.check_finite[m]):
                if not (torch.isfinite(current)):
                    print("Should stop for", m, "(not finite)")
                    should_stop_count += 1
                    continue
            monitor_op = self.mode_dict[self.mode[m]]
            print('current', current, 'best', self.best_score[m], 'monitor_op', monitor_op)
            
            if monitor_op(current - self.min_delta[m], self.best_score[m].to(current.device)):
                self.best_score[m] = current
                self.wait_count[m] = 0
            else:
                self.wait_count[m] += 1
                print('self.wait_count', self.wait_count)
                if self.wait_count[m] >= self.patience[m]:
                    print("Should stop for", m, " (not improved)")
                    should_stop_count += 1
        print("should_stop_count", should_stop_count)
        should_stop = should_stop_count == len(self.monitor)
        print('should_stop', should_stop)
        print('trainer.should_stop', trainer.should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        print('trainer.should_stop', trainer.should_stop)
        if should_stop:
            self.stopped_epoch = trainer.current_epoch'''
    
class MultiLossEarlyStopping(Callback):
    def __init__(self, monitor, min_delta, patience, mode, check_finite):
        super().__init__()
        self.mode_dict = {"min": torch.lt, "max": torch.gt}
        self.monitor = monitor
                
        self.min_delta = dict(zip(monitor, min_delta))
        self.patience = dict(zip(monitor, patience))
        self.mode = dict(zip(monitor, mode))
        self.check_finite = dict(zip(monitor, check_finite))
        
        self.best_score = {}
        self.wait_count = {}
        
        torch_inf = torch.tensor(torch.inf)
        
        for m in self.monitor:
            monitor_op = self.mode_dict[self.mode[m]]
            self.min_delta[m] *= 1 if monitor_op == torch.gt else -1
            self.best_score[m] = torch_inf if monitor_op == torch.lt else -torch_inf
            self.wait_count[m] = 0
    
    @override
    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        print('logs', logs, end='\n')
        
        should_stop_count = 0
        for m in self.monitor:
            current = logs[m].squeeze()
            if(self.check_finite[m]):
                if not (torch.isfinite(current)):
                    print("Should stop for", m, "(not finite)")
                    should_stop_count += 1
                    continue
            monitor_op = self.mode_dict[self.mode[m]]
            print('current', current, 'best', self.best_score[m], 'monitor_op', monitor_op)
            
            if monitor_op(current - self.min_delta[m], self.best_score[m].to(current.device)):
                self.best_score[m] = current
                self.wait_count[m] = 0
            else:
                self.wait_count[m] += 1
                print('self.wait_count', self.wait_count)
                if self.wait_count[m] >= self.patience[m]:
                    print("Should stop for", m, " (not improved)")
                    should_stop_count += 1
        print("should_stop_count", should_stop_count)
        should_stop = should_stop_count == len(self.monitor)
        print('should_stop', should_stop)
        print('trainer.should_stop', trainer.should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        print('trainer.should_stop', trainer.should_stop)
        if should_stop:
            self.stopped_epoch = trainer.current_epoch

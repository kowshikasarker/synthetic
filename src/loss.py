import torch
import torch.nn.functional as F

from pytorch_lightning.callbacks import Callback
from typing_extensions import override

class KLLoss(torch.nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    
    def forward(self, mean, logvar):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim = 1), dim = 0)
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

class Annealer(torch.nn.Module):
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        super(Annealer, self).__init__()
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = 'none'
            self.baseline = 0.0

    def forward(self, kl_loss):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        slope = self.slope()
        print('slope', slope, 'kl_loss', kl_loss.item())
        kl_loss = kl_loss * slope
        print('kl_loss', kl_loss.item())
        return slope, kl_loss

    def slope(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - self.current_step)
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        print('current_step', self.current_step)
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return
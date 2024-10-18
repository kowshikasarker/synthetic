import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv

class CombinedHiddenCVAEEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation):
        super(CombinedHiddenCVAEEncoder, self).__init__()
        self.i2h = Linear(input_dim, hidden_dim)
        self.mean = Linear(hidden_dim, latent_dim)
        self.logvar = Linear(hidden_dim, latent_dim)
        self.act = ReLU()

    def forward(self, x):
        h = self.act(self.i2h(x))
        mean = self.mean(h)
        logvar = self.logvar(h)

        noise = torch.randn(logvar.shape[0], logvar.shape[1])
        noise = noise.to(x.device)
        z = noise * torch.exp(0.5 * logvar) + mean # check eqn
        return z, mean, logvar
    
class CombinedHiddenCVAEDecoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super(CombinedHiddenCVAEDecoder, self).__init__()
        self.i2h = Linear(input_dim, hidden_dim)
        self.out = Linear(hidden_dim, output_dim)
        self.act = ReLU()

    def forward(self, x):
        h = self.act(self.i2h(x))
        out = self.out(h)
        return out
    
class CombinedHiddenCVAE(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 condition_dim,
                 hidden_dim,
                 latent_dim,
                 activation):
        super(CombinedHiddenCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = CombinedHiddenCVAEEncoder(feature_dim+condition_dim,
                                                 hidden_dim,
                                                 latent_dim,
                                                 activation)
        self.decoder = CombinedHiddenCVAEDecoder(latent_dim + condition_dim,
                                                 hidden_dim,
                                                 feature_dim,
                                                 activation)
        
    def forward(self, feature, condition):
        x = torch.concat([feature, condition], dim=1)
        z, mean, logvar = self.encoder(x)
        x = torch.concat([z, condition], dim=1)
        out = self.decoder(x)
        return z, mean, logvar, out

    def sample(self, condition):
        z = torch.randn(condition.shape[0], self.latent_dim)
        z = z.to(condition.device)
        x = torch.concat([z, condition], dim=1)
        out = self.decoder(x)
        return z, out

class CombinedHiddenGCVAEEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation):
        super(CombinedHiddenGCVAEEncoder, self).__init__()
        self.i2h = GCNConv(input_dim, hidden_dim, add_self_loops=True)
        self.mean = GCNConv(hidden_dim, latent_dim, add_self_loops=True)
        self.logvar = GCNConv(hidden_dim, latent_dim, add_self_loops=True)
        self.act = ReLU()

    def forward(self, x, edge_index):
        h = self.act(self.i2h(x, edge_index))         
        mean = self.mean(h, edge_index)
        logvar = self.logvar(h, edge_index)

        noise = torch.randn(x.shape[0], logvar.shape[1])
        noise = noise.to(x.device)
        z = noise * torch.exp(0.5 * logvar) + mean # check eqn
        return z, mean, logvar

class CombinedHiddenGCVAEDecoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super(CombinedHiddenGCVAEDecoder, self).__init__()
        self.i2h = GCNConv(input_dim, hidden_dim, add_self_loops=True)
        self.out = GCNConv(hidden_dim, output_dim, add_self_loops=True)
        self.act = ReLU()

    def forward(self, x, edge_index):
        h = self.act(self.i2h(x, edge_index))
        out = self.out(h, edge_index)
        return out

class CombinedHiddenGCVAE(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 condition_dim,
                 hidden_dim,
                 latent_dim,
                 activation):
        super(CombinedHiddenGCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = CombinedHiddenGCVAEEncoder(feature_dim + condition_dim,
                                                  hidden_dim,
                                                  latent_dim,
                                                  activation)
        self.decoder = CombinedHiddenGCVAEDecoder(latent_dim + condition_dim,
                                                  hidden_dim,
                                                  feature_dim,
                                                  activation)
        
    def forward(self, feature, condition, edge_index):
        x = torch.concat([feature, condition], dim=1)
        z, mean, logvar = self.encoder(x, edge_index)
        x = torch.concat([z, condition], dim=1)
        out = self.decoder(x, edge_index)
        return z, mean, logvar, out

    def sample(self, condition, edge_index):
        z = torch.randn(condition.shape[0], self.latent_dim)
        z = z.to(condition.device)
        x = torch.concat([z, condition], dim=1)
        out = self.decoder(x, edge_index)
        return z, out
        
    
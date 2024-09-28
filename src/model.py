import torch
from torch_geometric.nn import GCNConv

class SeparateHiddenEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, self_loop):
        super(SeparateHiddenEncoder, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim, add_self_loops=self_loop) # self_loop -> true/false
        self.mean = GCNConv(hidden_dim, latent_dim, add_self_loops=self_loop)
        self.logvar = GCNConv(hidden_dim, latent_dim, add_self_loops=self_loop)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        mean = self.mean(h, edge_index)
        logvar = self.logvar(h, edge_index)

        noise = torch.randn(x.shape[0], logvar.shape[1])
        noise = noise.to(x.device)
        z = noise * torch.exp(0.5 * logvar) + mean # check eqn
        return z, mean, logvar

class SeparateHiddenDecoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, self_loop):
        super(SeparateHiddenDecoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=self_loop)
        self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=self_loop)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        out = self.conv2(h, edge_index)
        return out

class SeparateHiddenModel(torch.nn.Module):
    def __init__(self, feature_dim, condition_dim, hidden_dim, latent_dim, self_loop):
        super(SeparateHiddenModel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = SeparateHiddenEncoder(feature_dim + condition_dim, hidden_dim, latent_dim, self_loop)
        self.decoder = SeparateHiddenDecoder(latent_dim + condition_dim, hidden_dim, feature_dim, self_loop)
        
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
    

class CombinedHiddenEncoder(torch.nn.Module):
    def __init__(self, feature_dim, condition_dim, hidden_dim, latent_dim, self_loop):
        super(CombinedHiddenEncoder, self).__init__()
        self.conv1 = GCNConv(feature_dim, hidden_dim, add_self_loops=self_loop) # self_loop -> true/false
        self.conv2 = GCNConv(condition_dim, hidden_dim, add_self_loops=self_loop)
        self.conv3 = GCNConv(2*hidden_dim, hidden_dim, add_self_loops=self_loop)
        self.mean = GCNConv(hidden_dim, latent_dim, add_self_loops=self_loop)
        self.logvar = GCNConv(hidden_dim, latent_dim, add_self_loops=self_loop)

    def forward(self, feature, condition, edge_index):
        f2h = self.conv1(feature, edge_index)
        c2h = self.conv2(condition, edge_index)
        h = torch.concat([f2h, c2h], dim=1)
        h = self.conv3(h, edge_index)
        mean = self.mean(h, edge_index)
        logvar = self.logvar(h, edge_index)
        noise = torch.randn(feature.shape[0], logvar.shape[1])
        noise = noise.to(feature.device)
        z = noise * torch.exp(0.5 * logvar) + mean # check eqn
        return z, mean, logvar

class CombinedHiddenDecoder(torch.nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim, self_loop):
        super(CombinedHiddenDecoder, self).__init__()
        self.conv1 = GCNConv(latent_dim, hidden_dim, add_self_loops=self_loop)
        self.conv2 = GCNConv(condition_dim, hidden_dim, add_self_loops=self_loop)
        self.conv3 = GCNConv(2*hidden_dim, hidden_dim, add_self_loops=self_loop)
        self.conv4 = GCNConv(hidden_dim, output_dim, add_self_loops=self_loop)

    def forward(self, latent, condition, edge_index):
        z2h = self.conv1(latent, edge_index)
        c2h = self.conv2(condition, edge_index)
        h = torch.concat([z2h, c2h], dim=1)
        h = self.conv3(h, edge_index)
        out = self.conv4(h, edge_index)
        return out

class CombinedHiddenModel(torch.nn.Module):
    def __init__(self, feature_dim, condition_dim, hidden_dim, latent_dim, self_loop):
        super(CombinedHiddenModel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = CombinedHiddenEncoder(feature_dim, condition_dim, hidden_dim, latent_dim, self_loop)
        self.decoder = CombinedHiddenDecoder(latent_dim, condition_dim, hidden_dim, feature_dim, self_loop)
        
    def forward(self, feature, condition, edge_index):
        z, mean, logvar = self.encoder(feature, condition, edge_index)
        out = self.decoder(z, condition, edge_index)
        return z, mean, logvar, out
    
    def sample(self, condition, edge_index):
        z = torch.randn(condition.shape[0], self.latent_dim)
        z = z.to(condition.device)
        x = torch.concat([z, condition], dim=1)
        print('x', x.shape)
        out = self.decoder(x, edge_index)
        return z, out
    
class UnconditionalEncoder(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim, self_loop):
        super(UnconditionalEncoder, self).__init__()
        self.conv = GCNConv(feature_dim, hidden_dim, add_self_loops=self_loop) # self_loop -> true/false
        self.mean = GCNConv(hidden_dim, latent_dim, add_self_loops=self_loop)
        self.logvar = GCNConv(hidden_dim, latent_dim, add_self_loops=self_loop)

    def forward(self, feature, edge_index):
        h = self.conv(feature, edge_index)
        mean = self.mean(h, edge_index)
        logvar = self.logvar(h, edge_index)
        noise = torch.randn(feature.shape[0], logvar.shape[1])
        noise = noise.to(feature.device)
        z = noise * torch.exp(0.5 * logvar) + mean # check eqn
        return z, mean, logvar

class UnconditionalDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, self_loop):
        super(UnconditionalDecoder, self).__init__()
        self.conv1 = GCNConv(latent_dim, hidden_dim, add_self_loops=self_loop)
        self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=self_loop)

    def forward(self, latent, edge_index):
        h = self.conv1(latent, edge_index)
        out = self.conv2(h, edge_index)
        return out
    
class UnconditionalModel(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim, self_loop):
        super(UnconditionalModel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = UnconditionalEncoder(feature_dim, hidden_dim, latent_dim, self_loop)
        self.decoder = UnconditionalDecoder(latent_dim, hidden_dim, feature_dim, self_loop)
        
    def forward(self, feature, edge_index):
        z, mean, logvar = self.encoder(feature, edge_index)
        out = self.decoder(z, edge_index)
        return z, mean, logvar, out
    
    def sample(self, sample_count, edge_index):
        z = torch.randn(sample_count, self.latent_dim)
        z = z.to(edge_index.device)
        out = self.decoder(z, edge_index)
        return z, out
        
    

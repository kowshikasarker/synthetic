import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class SeparateHiddenCVAE(torch.nn.Module):
    def __init__(self, feature_dim, condition_dim, hidden_dim, latent_dim):
        super(SeparateHiddenCVAE, self).__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # encode
        self.f2h = Linear(feature_dim, hidden_dim)
        self.c2h_e = Linear(condition_dim, hidden_dim)
        self.mean = Linear(2*hidden_dim, latent_dim)
        self.logvar = Linear(2*hidden_dim, latent_dim)

        # decode
        self.z2h = Linear(latent_dim, hidden_dim)
        self.c2h_d = Linear(condition_dim, hidden_dim)
        self.out = Linear(2*hidden_dim, feature_dim)

    def encode(self, feature, condition):
        #print()
        #print('SeparateHiddenCVAE encode')
        f2h = self.f2h(feature).tanh()
        c2h = self.c2h_e(condition).tanh()
        h = torch.cat([f2h, c2h], dim=1) 
        mean = self.mean(h)
        logvar = self.logvar(h)
        noise = torch.randn(logvar.shape[0], logvar.shape[1])
        noise = noise.to(feature.device)
        z = noise * torch.exp(0.5 * logvar) + mean
        return z, mean, logvar

    def decode(self, z, condition):
        z2h = self.z2h(z).tanh()
        c2h = self.c2h_d(condition).tanh()
        h = torch.cat([z2h, c2h], dim=1)
        out = self.out(h)
        return out

    def forward(self, feature, condition):
        z, mean, logvar = self.encode(feature, condition)
        out = self.decode(z, condition)
        return z, mean, logvar, out
    
    def sample(self, condition):
        z = torch.randn(condition.shape[0], self.latent_dim)
        z = z.to(condition.device)
        out = self.decode(z, condition)
        return z, out
    
class CombinedHiddenCVAE(torch.nn.Module):
    def __init__(self, feature_dim, condition_dim, hidden_dim, latent_dim):
        super(CombinedHiddenCVAE, self).__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # encode
        self.h_e = Linear(feature_dim+condition_dim, hidden_dim)
        self.mean = Linear(hidden_dim, latent_dim)
        self.logvar = Linear(hidden_dim, latent_dim)

        # decode
        self.h_d = Linear(latent_dim+condition_dim, hidden_dim)
        self.out = Linear(hidden_dim, feature_dim)

    def encode(self, x): # Q(z|x, c)
        h = self.h_e(x).tanh() 
        mean = self.mean(h)
        logvar = self.logvar(h)
        noise = torch.randn(logvar.shape[0], logvar.shape[1])
        noise = noise.to(x.device)
        z = noise * torch.exp(0.5 * logvar) + mean
        return z, mean, logvar

    def decode(self, x):
        h = self.h_d(x).tanh()
        out = self.out(h)
        return out

    def forward(self, feature, condition):
        x = torch.cat([feature, condition], dim=1)
        z, mean, logvar = self.encode(x)
        x = torch.cat([z, condition], dim=1)
        out = self.decode(x)
        return z, mean, logvar, out
    
    def sample(self, condition):
        z = torch.randn(condition.shape[0], self.latent_dim)
        z = z.to(condition.device)
        x = torch.cat([z, condition], dim=1)
        out = self.decode(x)
        return z, out

class CombinedHiddenPRADAEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(CombinedHiddenPRADAEncoder, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim, add_self_loops=True) # self_loop -> true/false
        self.mean = GCNConv(hidden_dim, latent_dim, add_self_loops=True)
        self.logvar = GCNConv(hidden_dim, latent_dim, add_self_loops=True)

    def forward(self, x, edge_index):
        print()
        print('CombinedHiddenPRADAEncoder')
        h = self.conv(x, edge_index).tanh()
        mean = self.mean(h, edge_index)
        logvar = self.logvar(h, edge_index)

        noise = torch.randn(x.shape[0], logvar.shape[1])
        noise = noise.to(x.device)
        z = noise * torch.exp(0.5 * logvar) + mean # check eqn
        
        '''exp = torch.exp(0.5 * logvar)
        nan = torch.isnan(exp).int()
        nan_cnt = nan.sum()
        inf = torch.isinf(exp).int()
        inf_cnt = inf.sum()
        print('nan_cnt', nan_cnt, 'inf_cnt', inf_cnt)
        if(nan_cnt > 0):
            nan_idx = torch.nonzero(nan)
            print('nan_idx logvar', logvar[nan_idx])
        if(inf_cnt > 0):
            inf_idx = torch.nonzero(inf)
            print('inf_idx logvar', logvar[inf_idx])
            
        print('x', torch.isnan(x).sum(), torch.isinf(x).sum())
        print('edge_index', torch.isnan(edge_index).sum(), torch.isinf(edge_index).sum())
        print('h', torch.isnan(h).sum(), torch.isinf(h).sum())
        print('mean', torch.isnan(mean).sum(), torch.isinf(mean).sum())
        print('logvar', torch.isnan(logvar).sum(), torch.isinf(logvar).sum())
        print('noise', torch.isnan(noise).sum(), torch.isinf(noise).sum())
        print('z', torch.isnan(z).sum(), torch.isinf(z).sum())
        print()'''
        return z, mean, logvar

class CombinedHiddenPRADADecoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CombinedHiddenPRADADecoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=True)

    def forward(self, x, edge_index):
        print()
        print('CombinedHiddenPRADADecoder')
        h = self.conv1(x, edge_index).tanh()
        out = self.conv2(h, edge_index)
        
        '''print('x', torch.isnan(x).sum(), torch.isinf(x).sum())
        print('edge_index', torch.isnan(edge_index).sum(), torch.isinf(edge_index).sum())
        print('h', torch.isnan(h).sum(), torch.isinf(h).sum())
        print('out', torch.isnan(out).sum(), torch.isinf(out).sum())
        print()'''
        return out

class CombinedHiddenPRADA(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 condition_dim,
                 hidden_dim,
                 latent_dim):
        super(CombinedHiddenPRADA, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = CombinedHiddenPRADAEncoder(feature_dim + condition_dim,
                                                  hidden_dim,
                                                  latent_dim)
        self.decoder = CombinedHiddenPRADADecoder(latent_dim + condition_dim,
                                                  hidden_dim,
                                                  feature_dim)
        
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
    

class SeparateHiddenPRADAEncoder(torch.nn.Module):
    def __init__(self, feature_dim, condition_dim, hidden_dim, latent_dim):
        super(SeparateHiddenPRADAEncoder, self).__init__()
        self.f2h = GCNConv(feature_dim, hidden_dim, add_self_loops=True) # self_loop -> true/false
        self.c2h = GCNConv(condition_dim, hidden_dim, add_self_loops=True)
        self.mean = GCNConv(2*hidden_dim, latent_dim, add_self_loops=True)
        self.logvar = GCNConv(2*hidden_dim, latent_dim, add_self_loops=True)

    def forward(self, feature, condition, edge_index):
        print('SeparateHiddenEncoder', 'edge_index', edge_index.dtype)
        f2h = self.f2h(feature, edge_index).tanh()
        c2h = self.c2h(condition, edge_index).tanh()
        h = torch.concat([f2h, c2h], dim=1)
        mean = self.mean(h, edge_index)
        logvar = self.logvar(h, edge_index)
        noise = torch.randn(feature.shape[0], logvar.shape[1])
        noise = noise.to(feature.device)
        z = noise * torch.exp(0.5 * logvar) + mean # check eqn
        return z, mean, logvar

class SeparateHiddenPRADADecoder(torch.nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim):
        super(SeparateHiddenPRADADecoder, self).__init__()
        self.z2h = GCNConv(latent_dim, hidden_dim, add_self_loops=True)
        self.c2h = GCNConv(condition_dim, hidden_dim, add_self_loops=True)
        self.out = GCNConv(2*hidden_dim, output_dim, add_self_loops=True)

    def forward(self, latent, condition, edge_index):
        z2h = self.z2h(latent, edge_index).tanh()
        c2h = self.c2h(condition, edge_index).tanh()
        h = torch.concat([z2h, c2h], dim=1)
        out = self.out(h, edge_index)
        return out

class SeparateHiddenPRADA(torch.nn.Module):
    def __init__(self, feature_dim, condition_dim, hidden_dim, latent_dim):
        super(SeparateHiddenPRADA, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = SeparateHiddenPRADAEncoder(feature_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = SeparateHiddenPRADADecoder(latent_dim, condition_dim, hidden_dim, feature_dim)
        
    def forward(self, feature, condition, edge_index):
        z, mean, logvar = self.encoder(feature, condition, edge_index)
        out = self.decoder(z, condition, edge_index)
        return z, mean, logvar, out
    
    def sample(self, condition, edge_index):
        z = torch.randn(condition.shape[0], self.latent_dim)
        z = z.to(condition.device)
        out = self.decoder(z, condition, edge_index)
        return z, out
    
class UnconditionalPRADAEncoder(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim):
        super(UnconditionalPRADAEncoder, self).__init__()
        self.conv = GCNConv(feature_dim, hidden_dim, add_self_loops=True) # self_loop -> true/false
        self.mean = GCNConv(hidden_dim, latent_dim, add_self_loops=True)
        self.logvar = GCNConv(hidden_dim, latent_dim, add_self_loops=True)

    def forward(self, feature, edge_index):
        h = self.conv(feature, edge_index).tanh()
        mean = self.mean(h, edge_index)
        logvar = self.logvar(h, edge_index)
        noise = torch.randn(feature.shape[0], logvar.shape[1])
        noise = noise.to(feature.device)
        z = noise * torch.exp(0.5 * logvar) + mean # check eqn
        return z, mean, logvar

class UnconditionalPRADADecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(UnconditionalPRADADecoder, self).__init__()
        self.conv1 = GCNConv(latent_dim, hidden_dim, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=True)

    def forward(self, latent, edge_index):
        h = self.conv1(latent, edge_index).tanh()
        out = self.conv2(h, edge_index)
        return out
    
class UnconditionalPRADA(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim):
        super(UnconditionalPRADA, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = UnconditionalPRADAEncoder(feature_dim, hidden_dim, latent_dim)
        self.decoder = UnconditionalPRADADecoder(latent_dim, hidden_dim, feature_dim)
        
    def forward(self, feature, edge_index):
        z, mean, logvar = self.encoder(feature, edge_index)
        out = self.decoder(z, edge_index)
        return z, mean, logvar, out
    
    def sample(self, sample_count, edge_index):
        z = torch.randn(sample_count, self.latent_dim)
        z = z.to(edge_index.device)
        out = self.decoder(z, edge_index)
        return z, out
        
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=0.1):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    @property
    def weight(self):
        epsilon = torch.randn_like(self.weight_mu)
        return self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * epsilon

    @property
    def bias(self):
        epsilon = torch.randn_like(self.bias_mu)
        return self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * epsilon

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def kl_divergence(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        kld_weight = (self.prior_sigma**2 + (self.weight_mu - self.prior_mu)**2) / (2 * weight_sigma**2) - 0.5
        kld_bias = (self.prior_sigma**2 + (self.bias_mu - self.prior_mu)**2) / (2 * bias_sigma**2) - 0.5
        return kld_weight.sum() + kld_bias.sum()

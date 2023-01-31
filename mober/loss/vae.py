import torch

from torch.distributions import Normal, kl_divergence
from torch.nn import functional


def loss_function_vae(dec, x, mu, stdev, kl_weight=1.0):
    # sum over genes, mean over samples, like trvae
    
    mean = torch.zeros_like(mu)
    scale = torch.ones_like(stdev)

    KLD = kl_divergence(Normal(mu, stdev), Normal(mean, scale)).mean(dim=1)

    reconst_loss = functional.mse_loss(dec, x, reduction='none').mean(dim=1)
    
    return (reconst_loss + kl_weight * KLD).sum(dim=0)
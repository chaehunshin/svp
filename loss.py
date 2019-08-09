import torch
import torch.nn.functional as F

def kl_div_loss(mu, logvar, mu_p, logvar_p):
    mu = mu.view(mu.shape[0], -1)
    logvar = logvar.view(logvar.shape[0], -1)
    sigma = torch.exp(0.5 * logvar)
    mu_p = mu_p.view(mu_p.shape[0], -1)
    logvar_p = logvar_p.view(logvar_p.shape[0], -1)
    sigma_p = torch.exp(0.5 * logvar_p)
    d_kl = torch.mean(torch.sum(torch.log(sigma_p/sigma) + (torch.exp(logvar) + (mu - mu_p)**2)/(2*torch.exp(logvar_p)) - 1/2, dim=-1))
    return d_kl


def img_recon_loss(recon_img, img):
    B = recon_img.shape[0]
    recon_img = recon_img.view(B, -1)
    img = img.view(B, -1)
    l2_loss = F.mse_loss(recon_img, img, size_average=False)/B
    return l2_loss



import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from layers import *

class SVP(nn.Module):
    def __init__(self, img_dim, h_dim, g_dim, latent_dim,
                 rnn_size, predictor_n_layers, posterior_n_layers, prior_n_layers):
        super().__init__()

        self._encoder = encoder(img_dim, h_dim)
        self._decoder = decoder(g_dim, img_dim)
        self._frame_predictor = lstm(h_dim + latent_dim, g_dim, rnn_size, predictor_n_layers)
        self._posterior = Gaussianlstm(h_dim, latent_dim, rnn_size, posterior_n_layers)
        self._prior = Gaussianlstm(h_dim, latent_dim, rnn_size, prior_n_layers)

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.xavier_normal_(m.weight, 2./math.sqrt(n))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                n = nn.init.calculate_gain('leaky_relu')
                nn.init.xavier_normal_(m.weight, n)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                n = m.in_features + m.out_features
                nn.init.xavier_normal_(m.weight, 2./math.sqrt(n))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return eps * torch.exp(0.5 * logvar) + mu

    # def prior_sample(self, ):
    def forward(self, prev_frame, curr_frame, posterior_state, prior_state, predictor_state, skip, phase=0):
        h = self._encoder(prev_frame)
        h_target, _ = self._encoder(curr_frame)
        if skip is None:
            h, skip = h
        else:
            h = h[0]
        mu, logvar, posterior_new_state = self._posterior(h_target, posterior_state)
        mu_p, logvar_p, prior_new_state = self._prior(h, prior_state)
        if phase == 0:
            z = self.reparameterize(mu, logvar)
        else:
            z = self.reparameterize(mu_p, logvar_p)
        # z = self.reparameterize(mu, logvar)

        h = torch.cat((h.squeeze(-1).squeeze(-1), z), dim=1)
        g, predictor_new_state = self._frame_predictor(h, predictor_state)
        predicted_frame = self._decoder(g, skip)

        return predicted_frame, mu, logvar, mu_p, logvar_p, posterior_new_state, prior_new_state, predictor_new_state, skip

    def inference(self, prev_frame, prior_state, predictor_state, skip):
        h = self._encoder(prev_frame)
        if skip is None:
            h, skip = h
        else:
            h = h[0]
        mu_p, logvar_p, prior_new_state = self._prior(h, prior_state)
        z = self.reparameterize(mu_p, logvar_p)

        h = torch.cat((h.squeeze(-1).squeeze(-1), z), dim=1)
        g, predictor_new_state = self._frame_predictor(h, predictor_state)
        predicted_frames = self._decoder(g, skip)

        return predicted_frames, prior_new_state, predictor_new_state, skip

    def posterior_initial_update(self, initial_frame):
        h, _ = self._encoder(initial_frame)
        _, _, posterior_new_state = self._posterior(h, None)
        return posterior_new_state

    def save(self, save_path, epoch, step):
        ckpt = {}
        ckpt['model'] = self.state_dict()
        ckpt['epoch'] = epoch
        ckpt['step'] = step
        torch.save(ckpt, save_path)

    def load(self, load_path):
        ckpt = torch.load(load_path)
        self.load_state_dict(ckpt['model'])
        epoch = ckpt['epoch']
        step = ckpt['step']

        return epoch, step

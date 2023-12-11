import math
import torch
import numpy as np
from torch import nn
from torch.autograd import grad
from functorch import vmap
from typing import List

from .utils import (
    config_activation, gaussian_repar, gaussian_log_prob,
    MLP,
)
from .energy_tensor_cnce import _EnergyTDBase, GaussianFourierProjection


INIT_QZ_SIGMA = 0.2


class EnergyTDTime(_EnergyTDBase):
    def __init__(
        self,
        tensor_shape: List[int],
        rank: int,
        h_dim: List[int],
        act: str = 'elu',
        dropout: float = 0.,
        latent_dim: int = 20,
        embedding_size: int = 8,
        nu: int = 20,
        sigma_func: str = 'exp',
        noise_sigma: float = 1.,
        pooling_method: str = 'sum',
        skip_connection: bool = False,
    ):
        super(EnergyTDTime, self).__init__(
            tensor_shape=tensor_shape,
            rank=rank,
            h_dim=h_dim,
            act=act,
            dropout=dropout,
            latent_dim=latent_dim,
            embedding_size=embedding_size,
            nu=nu,
            sigma_func=sigma_func,
            pooling_method=pooling_method,
            skip_connection=skip_connection
        )
        self.noise_sigma = nn.Parameter(
            torch.tensor(noise_sigma), requires_grad=False)
        self.setup_energy_()
        self.time_embedding = GaussianFourierProjection(4)
        # act = config_activation(self.act)
        # self.time_embedding = MLP(
        #     input_dim=1, output_dim=8,
        #     h_dim=[20, 20], act=act, dropout=self.dropout,
        #     bn=False, wn=False, sn=False, skip_connection=self.skip_connection)

    def setup_energy_(self):
        act = config_activation(self.act)
        # z encoder
        z_enc = MLP(
            input_dim=self.dim * self.rank, output_dim=self.latent_dim,
            h_dim=self.h_dim, act=act, dropout=self.dropout,
            bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # x encoder
        emb_size = 9
        x_enc = MLP(
            input_dim=emb_size, output_dim=self.latent_dim,
            h_dim=self.h_dim, act=act, dropout=self.dropout,
            bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # ouput layer
        if self.pooling_method in ['sum', 'attn']:
            in_size = self.latent_dim
        elif self.pooling_method in ['cat', 'sum_cat']:
            in_size = 2 * self.latent_dim
        else:
            raise RuntimeError('Wrong pooling method!')
        output_layer = MLP(
            input_dim=in_size, output_dim=int(in_size / 2),
            h_dim=self.h_dim, act=act, dropout=self.dropout,
            bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        self.layers = nn.ModuleDict({
            'z_enc': z_enc, 'x_enc': x_enc, 'output': output_layer
        })

    def forward(self, idx, t, x, sample=True, return_z=False):
        z = []
        for d in range(self.dim):
            if sample:
                z_d = gaussian_repar(
                    mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))
            else:
                z_d = self.q_z_mu[d]
            z.append(z_d[idx[:, d]])
        z_ten = torch.cat(z, -1)

        # expand input
        t = self.time_embedding(t)
        # x_exp = torch.stack([x, t], -1)
        x_exp = torch.cat([x.view(-1, 1), t], -1)
        x_exp = self.layers['x_enc'](x_exp)
        z_exp = self.layers['z_enc'](z_ten)

        if self.pooling_method == 'sum':
            hidden = z_exp + x_exp
        elif self.pooling_method == 'attn':
            hidden = torch.sigmoid(z_exp) * x_exp
        elif self.pooling_method == 'cat':
            hidden = torch.cat([z_exp, x_exp], -1)
        elif self.pooling_method == 'sum_cat':
            hidden = torch.cat([z_exp + x_exp, x_exp], -1)
        else:
            raise RuntimeError('Wrong pooling method!')
        energy = self.layers['output'](hidden).pow(2).sum(-1)

        if return_z:
            return - energy, z
        else:
            return - energy

    def loss(self, idx, t, x):
        # conditional variational noise contrastive estimation
        assert isinstance(self.nu, int)
        # first term
        phi_xz, z_x = self.forward(idx, t, x, return_z=True, sample=True)
        log_q_z_x = 0.
        for d in range(self.dim):
            q_z_mu = self.q_z_mu[d][idx[:, d]]
            q_z_sigma = self.q_z_sigma(d)[idx[:, d]]
            log_q_z_x = log_q_z_x + gaussian_log_prob(
                x=z_x[d], x_mu=q_z_mu, x_sigma=q_z_sigma
            ).sum(-1)
        log_weighted_phi_xz = phi_xz - log_q_z_x  # batch

        # second term
        log_weighted_phi_yz = []
        for _ in range(self.nu):
            y = torch.randn_like(x) * self.noise_sigma + x
            phi_yz, z_y = self.forward(idx, t, y, return_z=True)
            log_q_z_y = 0.
            for d in range(self.dim):
                q_z_mu = self.q_z_mu[d][idx[:, d]]
                q_z_sigma = self.q_z_sigma(d)[idx[:, d]]
                log_q_z_y = log_q_z_y + gaussian_log_prob(
                    x=z_y[d], x_mu=q_z_mu, x_sigma=q_z_sigma
                ).sum(-1)
            log_weighted_phi_yz.append(phi_yz - log_q_z_y)

        loss = 0.
        for i in range(self.nu):
            r = log_weighted_phi_yz[i] - log_weighted_phi_xz
            loss = loss + nn.functional.softplus(r).mean()
        return - loss * 2. / self.nu

    def langevin_sample_posterior(
        self, idx, time, x0=None, step=1000, epsilon=.02, t=1.
    ):
        def energy(x):
            log_prob = self.forward(idx, time, x, sample=False, return_z=False)
            return log_prob.sum()

        dev = self.q_z_mu[0].device
        dtype = self.q_z_mu[0].dtype
        if x0 is None:
            x0 = torch.rand(idx.shape[0], device=dev, dtype=dtype)
        x0.requires_grad_(True)
        for _ in range(step):
            score = grad(energy(x0), x0)[0]
            x0.data = x0.data + 0.5 * epsilon ** 2 * score \
                + epsilon * math.sqrt(t) * torch.randn_like(score)
        return x0

    def grid_search_posterior(self, idx, t, x_range, epsilon):
        assert x_range is not None, "Please specify sample range!"
        dev = self.q_z_mu[0].device
        dtype = self.q_z_mu[0].dtype
        x_grid = torch.arange(
            start=x_range[0], end=x_range[1] + epsilon / 2., step=epsilon,
            device=dev, dtype=dtype
        ).view(-1, 1).repeat(1, len(idx))  # Grid x Batch

        def unnorm_log_prob(x):
            out = self.forward(idx, t, x, sample=False, return_z=False)
            return out

        u_log_prob = vmap(unnorm_log_prob, in_dims=0)(x_grid)
        idx_est = torch.argmax(u_log_prob, dim=0)
        x_opt = torch.gather(x_grid, 0, idx_est.unsqueeze(0)).squeeze()

        return x_opt

    def predict(
        self,
        idx=None,
        time=None,
        batch=1e4,
        sampling=False,
        x0=None,
        step=100,
        epsilon=0.02,  # learning rate or grid interval
        t=1.,
        x_range=None,
    ):
        if idx is not None:
            idx_array = [idx]
            idx_list = None
            idx_full = None
        else:
            idx_array, idx_list, idx_full = self._split_idx(batch)

        tensor_predict = []
        chunk_num = len(idx_array)
        for n in range(chunk_num):

            if sampling:  # use Langevin Dynamics sampling
                if x0 is None:
                    x0_n = None
                else:
                    x0_n = x0[idx_list[n]]

                x_hat = self.langevin_sample_posterior(
                    idx_array[n], x0=x0_n, step=step, epsilon=epsilon, t=t)
            else:  # use grid search for MAP
                x_hat = self.grid_search_posterior(
                    idx=idx_array[n], t=time, x_range=x_range, epsilon=epsilon)

            tensor_predict.append(x_hat)

        tensor_predict = torch.cat(tensor_predict).squeeze()
        if idx is not None:
            return tensor_predict

        device = self.q_z_mu[0].device
        dtype = self.q_z_mu[0].dtype
        tensor_full = torch.ones(
            self.tensor_shape, dtype=dtype, device=device)
        tensor_full[idx_full] = tensor_predict
        return tensor_full

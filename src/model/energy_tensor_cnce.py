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


INIT_QZ_SIGMA = 0.2


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""
    def __init__(self, embedding_size=32, scale=1.0):
        super(GaussianFourierProjection, self).__init__()
        self.W = nn.Parameter(
            torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class _EnergyTDBase(nn.Module):
    def __init__(
        self, tensor_shape, rank, h_dim, act, dropout,
        latent_dim, embedding_size, nu, sigma_func, pooling_method,
        skip_connection,
    ):
        super(_EnergyTDBase, self).__init__()
        self.tensor_shape = list(tensor_shape)
        self.rank = int(rank)
        self.h_dim = h_dim
        self.act = act
        self.dropout = dropout if dropout > 0. else None
        self.latent_dim = latent_dim
        self.embedding_size = embedding_size
        self.nu = int(nu)
        if sigma_func == 'exp':
            self.sigma_func = lambda x: torch.exp(x)
        elif sigma_func == 'softplus':
            self.sigma_func = lambda x: nn.functional.softplus(x)
        else:
            raise NotImplementedError
        self.pooling_method = pooling_method
        self.skip_connection = skip_connection

        self.dim = len(tensor_shape)

        self.setup_q_z_()
        self.setup_energy_()

    def q_z_sigma(self, d):
        return self.sigma_func(self.q_z_log_sigma[d])

    def setup_q_z_(self):
        q_z_mu = []
        q_z_log_sigma = []
        for s in self.tensor_shape:
            q_z_mu.append(nn.Parameter(torch.empty(s, self.rank)))
            q_z_log_sigma.append(nn.Parameter(torch.empty(s, self.rank)))

        self.q_z_mu = nn.ParameterList(q_z_mu)
        self.q_z_log_sigma = nn.ParameterList(q_z_log_sigma)

        for q in self.q_z_mu:
            torch.nn.init.normal_(q.data, 0., INIT_QZ_SIGMA)
        for q in self.q_z_log_sigma:
            torch.nn.init.normal_(q.data, -2., INIT_QZ_SIGMA)

    def setup_energy_(self):
        act = config_activation(self.act)
        # z encoder
        z_enc = MLP(
            input_dim=self.dim * self.rank, output_dim=self.latent_dim,
            h_dim=self.h_dim, act=act, dropout=self.dropout,
            bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # x encoder
        if hasattr(self, 'category'):
            if self.embedding_size is not None:
                emb_size = self.embedding_size
            else:
                emb_size = self.category
        else:
            emb_size = 1 if self.embedding_size == 1 \
                else 2 * self.embedding_size
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

    def _input_embedding(self, x):
        raise NotImplementedError

    def forward(self, idx, x, sample=True, return_z=False):
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
        x_exp = self._input_embedding(x)
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

    def predict(
        self,
        idx=None,
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
                    idx=idx_array[n], x_range=x_range, epsilon=epsilon)

            tensor_predict.append(x_hat)

        tensor_predict = torch.cat(tensor_predict).squeeze()
        if idx is not None:
            return tensor_predict

        # device = self.q_z_mu[0].device
        # dtype = self.q_z_mu[0].dtype
        # tensor_full = torch.ones(
        #     self.tensor_shape, dtype=dtype, device=device)
        # import ipdb; ipdb.set_trace()
        # tensor_full[idx_full] = tensor_predict
        tensor_full = tensor_predict.view(*self.tensor_shape)
        return tensor_full

    def _split_idx(self, batch):
        device = self.q_z_mu[0].device
        dtype = self.q_z_mu[0].dtype

        tensor_full = torch.ones(
            self.tensor_shape, dtype=dtype, device=device)
        idx = torch.nonzero(tensor_full)
        if idx.shape[0] < batch:
            chunk_num = 1
        else:
            chunk_num = int(idx.shape[0] / batch)
        idx_array = torch.chunk(idx, chunk_num)

        idx_list_ = torch.nonzero(tensor_full, as_tuple=True)
        idx_list_ = [torch.chunk(i, chunk_num) for i in idx_list_]
        idx_list = []
        for i in range(chunk_num):
            idx_cache = []
            for d in range(tensor_full.ndim):
                idx_cache.append(idx_list_[d][i])
            idx_list.append(tuple(idx_cache))

        return idx_array, idx_list, idx_list_

    def grid_search_posterior(self, idx, x_range, epsilon):
        raise NotImplementedError

    def langevin_sample_posterior(self, idx, x0, step, epsilon, t):
        raise NotImplementedError


class EnergyTDContinuous(_EnergyTDBase):
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
        super(EnergyTDContinuous, self).__init__(
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
        if embedding_size > 1:
            self.x_embedding = GaussianFourierProjection(embedding_size)
        else:
            self.register_module('x_embedding', None)

    def loss(self, idx, x):
        # conditional variational noise contrastive estimation
        assert isinstance(self.nu, int)
        # first term
        phi_xz, z_x = self.forward(idx, x, return_z=True, sample=True)
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
            phi_yz, z_y = self.forward(idx, y, return_z=True)
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

    def _input_embedding(self, x):
        if self.x_embedding is not None:
            x_exp = self.x_embedding(x)
        else:
            x_exp = x.view(-1, 1)
        return x_exp

    def langevin_sample_posterior(
        self, idx, x0=None, step=1000, epsilon=.02, t=1.
    ):
        def energy(x):
            log_prob = self.forward(idx, x, sample=False, return_z=False)
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

    def grid_search_posterior(self, idx, x_range, epsilon):
        assert x_range is not None, "Please specify sample range!"
        dev = self.q_z_mu[0].device
        dtype = self.q_z_mu[0].dtype
        x_grid = torch.arange(
            start=x_range[0], end=x_range[1] + epsilon / 2., step=epsilon,
            device=dev, dtype=dtype
        ).view(-1, 1).repeat(1, len(idx))  # Grid x Batch

        def unnorm_log_prob(x):
            out = self.forward(idx, x, sample=False, return_z=False)
            return out

        u_log_prob = vmap(unnorm_log_prob, in_dims=0)(x_grid)
        idx_est = torch.argmax(u_log_prob, dim=0)
        x_opt = torch.gather(x_grid, 0, idx_est.unsqueeze(0)).squeeze()

        return x_opt

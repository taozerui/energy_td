import math
import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm
from typing import Optional, List
from numbers import Real


def config_activation(act: Optional[str]):
    act_dict = {
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'selu': nn.SELU(),
        'silu': nn.SiLU()
    }
    if act is None:
        return None
    else:
        assert act.lower() in act_dict.keys(), "ACT not supported"
        return act_dict[act.lower()]


def gaussian_repar(mu, sigma):
    noise = torch.randn_like(mu) * sigma
    return mu + noise


def gaussian_log_prob(x, x_mu, x_sigma):
    if isinstance(x_sigma, Real):
        log_x_sigma = math.log(x_sigma)
    else:
        log_x_sigma = torch.log(x_sigma)

    log_prob = - 0.5 * math.log(2 * math.pi) - log_x_sigma \
        - 0.5 * (torch.pow(x - x_mu, 2) / x_sigma ** 2)
    return log_prob


def setup_mlp(
    input_dim: int, output_dim: int, h_dim: List[int],
    act: Optional[nn.Module], out_act: Optional[nn.Module],
    dropout: Optional[float] = None,
    bn: bool = False, wn: bool = False, sn: bool = False
):
    layer = []
    layer_dim = [input_dim] + h_dim
    if dropout is not None:
        dropout_layer = nn.Dropout(dropout)
    for i in range(len(layer_dim) - 1):
        if wn:
            layer.append(
                weight_norm(nn.Linear(layer_dim[i], layer_dim[i+1]))
            )
        elif sn:
            layer.append(
                spectral_norm(nn.Linear(layer_dim[i], layer_dim[i+1]))
            )
        else:
            layer.append(
                nn.Linear(layer_dim[i], layer_dim[i+1])
            )
        if act is not None:
            layer.append(act)
        if bn:
            layer.append(
                nn.BatchNorm1d(layer_dim[i+1], track_running_stats=False))
        if dropout is not None:
            layer.append(dropout_layer)
    if wn:
        layer.append(
            weight_norm(nn.Linear(layer_dim[-1], output_dim))
        )
    elif sn:
        layer.append(
            spectral_norm(nn.Linear(layer_dim[-1], output_dim))
        )
    else:
        layer.append(
            nn.Linear(layer_dim[-1], output_dim)
        )
    if out_act is not None:
        layer.append(out_act)

    return nn.Sequential(*layer)


class MLP(nn.Module):
    """docstring for MLP."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        h_dim: List[int],
        act: Optional[nn.Module],
        skip_connection: bool = True,
        dropout: Optional[float] = None,
        bn: bool = False,
        wn: bool = False,
        sn: bool = False,
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.h_dim = h_dim
        self.act = act
        self.bn = bn
        self.wn = wn
        self.sn = sn
        self.skip_connection = skip_connection
        self.dropout = dropout

        self.layer = setup_mlp(
            input_dim, output_dim, h_dim, self.act, out_act=None,
            dropout=dropout, bn=bn, wn=wn, sn=sn)
        if skip_connection:
            self.skip_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.layer(x)
        if self.skip_connection:
            out = out + self.skip_layer(x)
        return out

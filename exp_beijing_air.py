import os
import shutil
import json
import time
import torch
import argparse
import numpy as np
import yaml
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

from src.model import EnergyTDTime
from src.data import get_air_data


class Trainer:
    """docstring for Trainer."""
    def __init__(
        self,
        model,
        conf,
        optimizer,
        print_eval=True,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.conf = conf
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=conf.train.lr
        ) if optimizer is None else optimizer
        miles = [int(i * conf.train.epoch) for i in conf.train.mile_stones]
        self.miles = miles
        self.opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer, milestones=miles, gamma=0.3
        )

        self.print_eval = print_eval
        self.eval_interval = conf.train.eval_int

        self.current_epoch = 0
        self.current_iter = 0

        self.log_test_metric = {'RMSE': [], 'MAE': [], 'MAPE': []}

    def train(self, train_loader, valid_loader=None, test_loader=None):
        bar = tqdm(range(self.conf.train.epoch), desc='[Epoch 0]')
        for epoch in bar:
            bar.set_description(f'[Epoch {epoch}]')
            self.train_epoch(train_loader)
            self.opt_scheduler.step()
            bar.set_postfix({'Loss': self.current_loss})

            is_eval = epoch % self.eval_interval == 0 or \
                epoch == self.conf.train.epoch - 1

            if is_eval:
                if valid_loader is not None:
                    self.eval_epoch(valid_loader, 'Valid')
                if test_loader is not None:
                    self.eval_epoch(test_loader, 'Test')

            self.current_epoch += 1

    def train_epoch(self, data_loader):
        model = self.model
        model.train()

        loss_log = []
        bar = tqdm(data_loader, desc='[Iter 0]', leave=False)
        for batch_idx, (inputs, x_time, x_val) in enumerate(bar):
            if torch.cuda.is_available():
                inputs, x_val = inputs.cuda(), x_val.cuda()
                x_time = x_time.cuda()

            if hasattr(self.conf.train, 'data_noise'):
                if hasattr(self.conf.model, 'category'):
                    x_val_ = x_val.clone().float()
                    x_val_[x_val_ == 1.0] = 1.0 - self.conf.train.data_noise
                    x_val_[x_val_ == 0.0] = self.conf.train.data_noise
                    torch.bernoulli(x_val_, out=x_val)
                else:
                    x_val += torch.randn_like(x_val) * self.conf.train.data_noise

            vnce = model.loss(inputs, x_time, x_val)
            loss = - vnce
            if batch_idx % 10:
                bar.set_postfix({'Loss': vnce.item()})
                bar.set_description(f'[Iter {batch_idx}]')

            self.optimizer.zero_grad()
            loss.backward()
            if hasattr(self.conf.train, 'grad_clip'):
                if self.current_epoch < self.miles[0] and self.conf.train.grad_clip > 0.:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.conf.train.grad_clip,
                        norm_type=self.conf.train.grad_clip_norm
                    )
            self.optimizer.step()

            if self.conf.train.log_grad_norm and writer is not None:
                gnorm_2 = gnorm_inf = 0.
                for param in model.parameters():
                    if param.grad is not None:
                        gnorm_2 += param.grad.data.norm(2).item() ** 2
                        gnorm_inf = np.maximum(param.grad.data.norm(torch.inf).item(), gnorm_inf)
                gnorm_2 = np.sqrt(gnorm_2)

            loss_log.append(vnce.item())
            self.current_iter += 1

        loss_log = np.mean(loss_log)
        self.current_loss = loss_log

    @torch.no_grad()
    def eval_epoch(self, test_loader, phase):
        self.scale = 5.
        model = self.model
        epoch = self.current_epoch

        model.eval()
        x_hat_tot = []
        x_val_tot = []
        for _, (inputs, x_time, x_val) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs, x_val = inputs.cuda(), x_val.cuda()
                x_time = x_time.cuda()

            x_hat = model.predict(
                inputs, x_time, x_range=[-0.5, 1.1], epsilon=1e-3).view(-1)
            x_hat_tot.append(x_hat * self.scale)
            x_val_tot.append(x_val * self.scale)

        x_hat_tot = torch.cat(x_hat_tot)
        x_val_tot = torch.cat(x_val_tot)

        rmse = torch.sqrt(torch.mean((x_hat_tot - x_val_tot).pow(2))).item()
        mae = torch.mean((x_hat_tot - x_val_tot).abs()).item()

        v = torch.clip(torch.abs(x_val_tot), 0.1, None)
        diff = torch.abs((x_val_tot - x_hat_tot) / v)
        mape = 100.0 * torch.mean(diff, axis=-1).mean().item()

        if self.print_eval:
            print(f'Epoch {epoch} - {phase}: RMSE is {rmse:.3f} | MAE is {mae:.3f}.')

        # if phase.lower() == 'test':
        self.log_test_metric['RMSE'].append(rmse)
        self.log_test_metric['MAE'].append(mae)
        self.log_test_metric['MAPE'].append(mape)


def main_run_func(args, conf, folds):
    # read data
    file_name = './dataset'
    data_loader = get_air_data(file_name, batch_size=conf.train.batch_size,)
    data_loader = data_loader[folds]

    # model
    model_conf = conf.model
    model = EnergyTDTime(
        tensor_shape=model_conf.tensor_shape,
        rank=args.rank,
        h_dim=model_conf.h_dim,
        act=model_conf.act,
        dropout=model_conf.dropout,
        latent_dim=model_conf.latent_dim,
        embedding_size=model_conf.embedding_size,
        nu=model_conf.nu,
        sigma_func=model_conf.sigma_func,
        noise_sigma=model_conf.noise_sigma,
        pooling_method=model_conf.pooling_method,
        skip_connection=model_conf.skip_connection
    )
    if torch.cuda.is_available():
        model = model.cuda()

    # trainer
    train_conf = conf.train
    optimizer = torch.optim.Adam(model.parameters(), lr=train_conf.lr)
    trainer = Trainer(
        model=model, conf=conf, optimizer=optimizer, print_eval=True)
    trainer.train(
        data_loader['train'], test_loader=data_loader['test'])

    with open(f'air_result_{folds}.txt', 'w') as file:
        file.write(json.dumps(trainer.log_test_metric))


def main():
    parser = argparse.ArgumentParser(description='Tensor completion')
    parser.add_argument('--rank', type=int, default=5,
                        choices=[3, 5, 8, 10])
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed')
    parser.add_argument('--debug', action='store_true', help='Debug')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # read config
    conf_path = './config/air.yaml'
    with open(conf_path) as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)

    # writer
    for i in range(5):
        main_run_func(args, conf, i)


if __name__ == "__main__":
    main()

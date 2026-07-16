#!/usr/bin/env python3
# maintained by kewei li
# -*- coding:utf-8 -*-


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR
import torch

def lr_scheduler(cfg, optimizer):

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1)

    if cfg.train.lr_scheduler.type == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.train.lr_scheduler.cosine.T_0,
            T_mult=cfg.train.lr_scheduler.cosine.T_mult,
            eta_min=cfg.train.lr_scheduler.cosine.eta_min)

    elif cfg.train.lr_scheduler.type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='max',
                                      factor=0.5,
                                      patience=1,
                                      threshold=0.0001,
                                      threshold_mode='rel',
                                      cooldown=0,
                                      min_lr=5e-5,
                                      eps=1e-08,
                                      verbose=False)

    return scheduler
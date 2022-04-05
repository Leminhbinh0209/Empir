#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:39:55 2022

@author: chingis
"""

import os
from backbones import get_model
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import wandb
from model_zoo import BYOL
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional
import torchvision.datasets as datasets
from pytorch_lightning.callbacks import Callback
import math
from easydict import EasyDict as edict
import torchmetrics
from utils_datasets import Databasket
# test model, a resnet 50

resnet = models.resnet50(pretrained=False)

DATASET = 'Flower'
# constants
args = edict(
    RUN = f'linear_{DATASET}_BYOL+JSD',
    BATCH_SIZE = 128,
    ACCUMULATE_GRAD_BATCHES = 1,
    EPOCHS     = 100,
    OVERFIT = 0,
    LR         = 3.2, #0.1,
    NUM_GPUS   = 2,
    IMAGE_SIZE = 224,
    NUM_WORKERS = 8,
    CLASSES = 102,
    ACCURACY = 'macro' # macro for Flower, other datasets micro according to BYOL paper
)
#custom Callbacks
class AdjustLearningRate(Callback):
    def __init__(self, init_lr, max_epoch, milestones = None, cos=False):
        super().__init__()
        self.epoch = 0
        self.milestones = milestones
        self.init_lr = init_lr
        self.cos = cos
        self.epochs = max_epoch
        self.current_lr = init_lr
        self.init_beta = 0.99
    def adjust_learning_rate(self, optimizer, pl_module, epoch):
        """Decay the learning rate based on schedule"""
        lr = self.init_lr
        beta = self.init_beta
        if self.cos:  # cosine lr schedule
            pl_module.learner.target_ema_updater.beta = 1 - (1 - beta) * (math.cos(math.pi * epoch / self.epochs) + 1) * 0.5
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / self.epochs))
        else:  # stepwise lr schedule
            assert self.milestones is not None
            for milestone in self.milestones:
                lr *= 0.1 if self.epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        self.current_lr = lr
    def on_train_epoch_start(self, trainer, pl_module):
        optimizers = trainer.optimizers
        for optimizer in optimizers:
            self.adjust_learning_rate(optimizer, pl_module, self.epoch)
        print('Current learning rate: ', self.current_lr)
        print('Current beta: ', pl_module.learner.target_ema_updater.beta)
    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch += 1

# pytorch lightning module
class TransferLearner(pl.LightningModule):
    def __init__(self, net, args, lr=0.1, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)
        self.lr = lr
        self.use_momentum = kwargs.get('use_momentum')
        self.fc = torch.nn.Linear(2048, args.CLASSES)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(average=args.ACCURACY,num_classes=args.CLASSES)
        self.valid_acc = torchmetrics.Accuracy(average=args.ACCURACY,num_classes=args.CLASSES)
        self.dt = None

    @torch.no_grad()
    def forward(self, images, target=None):
        self.learner.eval()
        return self.learner(images, target=target, return_embedding=True, return_projection=False)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        emb = self.forward(images)
        emb = emb.detach()
        pred = self.fc(emb)
        #print(pred.shape)
        loss = self.cross_entropy(pred, targets)

        self.train_acc(pred, targets)
        # Log training loss 
        self.log('training_loss', loss,  on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        emb = self.forward(images)
    
        pred = self.fc(emb)
        loss = self.cross_entropy(pred, targets)
        self.valid_acc(pred, targets)
        self.log('validaion_loss', loss,  on_step=True, on_epoch=True, prog_bar=True)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        sgd = torch.optim.SGD(self.fc.parameters(), lr=self.lr, momentum=0.9, weight_decay=0)
        from torchlars import LARS
        opt = LARS(sgd)
        opt.state = sgd.state
        return opt

# images dataset

# custom DataModule
class MyImageNetModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/home/data/", dataset='CIFAR10', num_workers=8, batch_size=128, image_size=225):
        super().__init__()
        self.data_dir = data_dir
        # according to BYOL do not use augmentations. Confirm
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.dataset = dataset
        self.num_workers = num_workers
        self.dataset_type = datasets.ImageFolder if dataset == 'Flower' else eval(f'datasets.{dataset}')
        self.batch_size = batch_size
    
    def prepare_data(self):
        # download
        if self.dataset != 'Flower':
            self.dataset_type(self.data_dir, train=True, download=True)
            self.dataset_type(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.dataset != 'Flower':
                self.train_dataset = self.dataset_type(self.data_dir, train=True, transform=self.train_transform)
                self.val_dataset = self.dataset_type(self.data_dir, train=False, transform=self.transform)
            else:
                self.train_dataset = self.dataset_type(f'{self.data_dir}/train', transform=self.train_transform)
                self.val_dataset = self.dataset_type(f'{self.data_dir}/test', transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_type(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.predict_dataset = self.dataset_type(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        print('Batch size: ', self.batch_size)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True, drop_last=False)

    def val_dataloader(self):
        print('Batch size: ', self.batch_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True, drop_last=False)


# main

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    
    # logger
    wandb_logger = WandbLogger(name=args.RUN, project=f'Self-Supervised Representation Learning FineTuning {DATASET}', config=args, entity="ssl2022")
    
    # DataModule
    dm = MyImageNetModule(
        dataset=DATASET,
        num_workers=args.NUM_WORKERS, 
        batch_size=args.BATCH_SIZE, 
        image_size=args.IMAGE_SIZE
    )
    PATH = 'results/byol+jsd.ckpt'
    init_lr= args.LR * args.BATCH_SIZE * args.NUM_GPUS / 256
    print(init_lr)
    model = TransferLearner(
        resnet,
        args,
        lr = init_lr,
        image_size = args.IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        use_momentum = False
        
    )
    
    # freeze all layers but the last fc
    for param in model.learner.parameters():
        param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    checkpoint = torch.load(PATH, map_location="cpu")
    if '.torch' in PATH:
        model.learner.online_encoder.net.load_state_dict(checkpoint)
    else:
         # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only learner.online_encoder or net
            if k.startswith('learner.online_encoder') or k.startswith('learner.net'):
                # remove prefix
                state_dict[k[len("learner."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        #print(state_dict.keys())
        model.learner.load_state_dict(state_dict, strict=False)
        # Callbacks

    lr_scheduler = AdjustLearningRate(
        init_lr=init_lr,
        max_epoch=args.EPOCHS,
        cos=True,
        
    )
    
    trainer = pl.Trainer(
        gpus = [0,1],
        strategy = 'ddp',
        max_epochs = args.EPOCHS,
        accumulate_grad_batches = args.ACCUMULATE_GRAD_BATCHES,
        sync_batchnorm = True,
        logger = wandb_logger,
        
        precision=16,
        callbacks=[lr_scheduler],
        #auto_scale_batch_size=True, Tuner used
        deterministic=True,
        # Testing purposes
        overfit_batches=args.OVERFIT,
    )

    trainer.fit(model, dm)

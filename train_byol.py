import os
from backbones import get_model
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import wandb
from byol_pytorch import BYOL, Experiment
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional
import torchvision.datasets as datasets
from pytorch_lightning.callbacks import Callback
import math
from easydict import EasyDict as edict
# test model, a resnet 50

OUTPUT_DIR = "../../../media/data1/binh/SSL/"

resnet = models.resnet50(pretrained=False, zero_init_residual=True)
# constants
args = edict(
    RUN = 'pretrain_imagenet_accGrad2_Byol',
    BATCH_SIZE = 128,
    ACCUMULATE_GRAD_BATCHES = 2,
    EPOCHS     = 300,
    OVERFIT = 0,
    LR         = 0.05,#1.2 * 1e-1,
    NUM_GPUS   = 2,
    IMAGE_SIZE = 224,
    NUM_WORKERS = 8,
    WARMUP = 10,
)
#custom Callbacks
class AdjustLearningRate(Callback):
    def __init__(self, init_lr, max_epoch, use_momentum=False, milestones = None, cos=False, warmup=0):
        super().__init__()
        self.epoch = 200
        self.milestones = milestones
        self.init_lr = init_lr
        self.cos = cos
        self.warmup = warmup
        self.epochs = 800 #max_epoch
        self.current_lr = init_lr
        self.init_beta = 0.99
        self.use_momentum = use_momentum
    def adjust_learning_rate(self, optimizer, pl_module, epoch):
        """Decay the learning rate based on schedule"""
        lr = self.init_lr
        beta = self.init_beta
        if self.cos:  # cosine lr schedule
            if self.use_momentum:
                pl_module.learner.target_ema_updater.beta = 1 - (1 - beta) * (math.cos(math.pi * epoch / self.epochs) + 1) * 0.5
            if epoch > self.warmup:
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
        if self.use_momentum:
            print('Current beta: ', pl_module.learner.target_ema_updater.beta)
    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch += 1

# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, lr=0.1, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)
        self.lr = lr
        self.use_momentum = kwargs.get('use_momentum')
    def forward(self, images, target=None):
        return self.learner(images, target=target)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        target = torch.arange(images.shape[0]).to(self.device)
        loss, top1, top5 = self.forward(images, target=target)
        if self.use_momentum:
            self.learner.update_moving_average() # update moving average of target encoder
        # Log training loss 
        self.log('training_loss', loss,  on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_top1', top1,  on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_top5', top5,  on_step=True, on_epoch=True, prog_bar=True)
        tqdm_dict = {'loss': loss.detach(), 'train_top1': top1.detach(), 'train_top5':top5.detach()}
        return {'loss': loss, 'progress_bar': tqdm_dict, 'log':tqdm_dict}

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        return opt
    # COMMENT: I think it should be updated every iteration
    #def on_before_zero_grad(self, _):
    #    if self.learner.use_momentum:
    #        self.learner.update_moving_average()

# images dataset

# custom DataModule
class MyImageNetModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/home/data/Imagenet", num_workers=8, batch_size=128, image_size=225):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.num_workers = num_workers
        self.dataset_type = datasets.ImageNet
        self.batch_size = batch_size
    '''
    def prepare_data(self):
        # download
        self.dataset_type(self.data_dir, train=True, download=True)
        self.dataset_type(self.data_dir, train=False, download=True)
    '''
    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_type(self.data_dir, split='train', transform=self.transform)
            #self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_type(self.data_dir, split='val', transform=self.transform)

        if stage == "predict" or stage is None:
            self.predict_dataset = self.dataset_type(self.data_dir, split='val', transform=self.transform)

    def train_dataloader(self):
        print('Batch size: ', self.batch_size)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True, drop_last=True)


# main

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    
    # logger
    wandb_logger = WandbLogger(name=args.RUN, 
                        project='Self-Supervised Representation Learning', 
                        config=args, logger=f"{OUTPUT_DIR}/byol/loggings/", 
                        entity="ssl2022")
    
    # DataModule
    dm = MyImageNetModule(
        num_workers=args.NUM_WORKERS, 
        batch_size=args.BATCH_SIZE, 
        image_size=args.IMAGE_SIZE
        )

    model = SelfSupervisedLearner(
        resnet,
        lr = args.LR * 4 * 128 / 256,
        image_size = args.IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        use_momentum = True,
        use_jsd = False,
        
    )
    # Callbacks
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor="train_top1",
        dirpath=f"{OUTPUT_DIR}/byol/results/",
        filename=args.RUN+'-{epoch:02d}-{train_top1:.2f}',
        save_last=True,
       # save_top_k=1,
       # mode="max",
        every_n_epochs = 100,
    )
    lr_scheduler = AdjustLearningRate(
        milestones=[1,2,3],
        init_lr=args.LR * 4 * 128 / 256,
        max_epoch=args.EPOCHS,
        cos=True,
        warmup=args.WARMUP
        
    )
    
    trainer = pl.Trainer(
        gpus = [0,1,2,3],
        strategy = 'ddp',
        max_epochs = args.EPOCHS,
        accumulate_grad_batches = args.ACCUMULATE_GRAD_BATCHES,
        sync_batchnorm = True,
        logger = wandb_logger,
        resume_from_checkpoint = f"{OUTPUT_DIR}/results/byol.ckpt",
        precision=16,
        callbacks=[checkpoint_callback, lr_scheduler],
        #auto_scale_batch_size=True, Tuner used
        deterministic=True,
        # Testing purposes
        overfit_batches=args.OVERFIT
    )

    trainer.fit(model, dm)

try:
    import pytorch_lightning
except:
    import sys
    sys.path.append("../pytorch-lightning/")
import os
from backbones import get_model
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import wandb
from model_zoo import MoCo, GaussianBlur, TwoCropsTransform
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

resnet = models.resnet50 # Moco (pretrained=False, zero_init_residual=True)
# constants
args = edict(
    RUN = 'pretrain_imagenet_accGrad2_MoCoV2',
    BATCH_SIZE = 64,
    ACCUMULATE_GRAD_BATCHES = 2,
    EPOCHS     = 300,
    OVERFIT = 0,
    LR         = 0.05,
    NUM_GPUS   = 2,
    IMAGE_SIZE = 224,
    NUM_WORKERS = 8,
    WARMUP = 10,
    # moco specific
    MOCO_K = 65536,
    MOCO_T = 0.07
)
#custom Callbacks
class AdjustLearningRate(Callback):
    def __init__(self, init_lr, max_epoch, use_momentum=False, milestones = None, cos=False, warmup=0):
        super().__init__()
        self.epoch = 0
        self.milestones = milestones
        self.init_lr = init_lr
        self.cos = cos
        self.warmup = warmup
        self.epochs = max_epoch
        self.current_lr = init_lr
        self.init_beta = 0.99
        self.use_momentum = use_momentum
    def adjust_learning_rate(self, optimizer, pl_module, epoch):
        """Decay the learning rate based on schedule"""
        lr = self.init_lr
        beta = self.init_beta
        if self.cos:  # cosine lr schedule
            if epoch > self.warmup:
                #epoch -= self.warmup
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
        self.learner = MoCo(net, **kwargs)
        self.lr = lr
    def forward(self, images):
        return self.learner(im_q=images[0], im_k=images[1])

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss, top1, top5 = self.forward(images)

        # Log training loss 
        self.log('training_loss', loss,  on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_top1', top1,  on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_top5', top5,  on_step=True, on_epoch=True, prog_bar=True)
        tqdm_dict = {'loss': loss.detach(), 'train_top1': top1.detach(), 'train_top5':top5.detach()}
        return {'loss': loss, 'progress_bar': tqdm_dict, 'log':tqdm_dict}

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        return opt


# images dataset

# custom DataModule
class MyImageNetModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/home/data/Imagenet", num_workers=8, batch_size=128, image_size=225):
        super().__init__()
        self.data_dir = data_dir
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
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
            self.train_dataset = self.dataset_type(self.data_dir, split='train', transform=TwoCropsTransform(self.transform))
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
                        config=args, save_dir=f"{OUTPUT_DIR}/byolv2/loggings/", 
                        entity="ssl2022")
    
    # DataModule
    dm = MyImageNetModule(
        num_workers=args.NUM_WORKERS, 
        batch_size=args.BATCH_SIZE, 
        image_size=args.IMAGE_SIZE
        )

    model = SelfSupervisedLearner(
        resnet,
        lr = args.LR * args.NUM_GPUS * args.BATCH_SIZE / 256,
        dim=256,
        K = args.MOCO_K,
        m = 0.999, 
        T = args.MOCO_T, 
        mlp = True # MoCov2
    )
    # Callbacks
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor="train_top1",
        dirpath=f"{OUTPUT_DIR}/byolv2/results/",
        filename=args.RUN+'-{epoch:02d}-{train_top1:.2f}',
        save_last=True,
        every_n_epochs = 100,
    )
    lr_scheduler = AdjustLearningRate(
        init_lr=args.LR * args.NUM_GPUS * args.BATCH_SIZE / 256,
        max_epoch=args.EPOCHS,
        cos=True,
        warmup=args.WARMUP
    )
    
    trainer = pl.Trainer(
        gpus = list(range(args.NUM_GPUS)),
        strategy = 'ddp',
        max_epochs = args.EPOCHS,
        accumulate_grad_batches = args.ACCUMULATE_GRAD_BATCHES,
        sync_batchnorm = True,
        logger = wandb_logger,
        resume_from_checkpoint = None,
        precision=16,
        callbacks=[checkpoint_callback, lr_scheduler],
        deterministic=True,
        overfit_batches=args.OVERFIT
    )

    trainer.fit(model, dm)

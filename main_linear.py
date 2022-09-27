# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from solo.utils.classification_dataloader import prepare_data
from solo.utils.checkpointer import Checkpointer
from solo.methods.linear import LinearModel
import types
from solo.utils.backbones import (
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)
from solo.methods.base import BaseMethod
from solo.args.setup import parse_args_linear
from torchvision.models import resnet18, resnet50
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
import torch.nn as nn
from torchvision import models
import torch
import os
import warnings
from time import sleep


try:
    from solo.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True


def main():
    args = parse_args_linear()

    assert args.backbone in BaseMethod._SUPPORTED_BACKBONES
    backbone_model = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
    }[args.backbone]

    # initialize backbone
    kwargs = args.backbone_args
    cifar = kwargs.pop("cifar", False)
    # swin specific
    if "swin" in args.backbone and cifar:
        kwargs["window_size"] = 4

    backbone = backbone_model(**kwargs)
    if "resnet" in args.backbone:
        # remove fc layer
        backbone.fc = nn.Identity()
        if cifar:
            backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    assert (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    )
    ckpt_path = args.pretrained_feature_extractor
    # print("########### VIC REG load model weights ##################")
    # state = torch.load(ckpt_path, map_location="cpu")
    # # VICReg
    # if "model" in state:
    #     state = state["model"]
    #     state = {
    #         key.replace("module.backbone.", ""): value
    #         for (key, value) in state.items()
    #     }

    # torch.save(dict(state_dict=state), ckpt_path.replace(
    #     "resnet50.pth", "sololearnstate_resnet50.pth"))
    # import sys
    # sys.exit(0)
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
            warnings.warn(
                "You are using an older checkpoint. Use a new one as some issues might arrise."
            )
            del state[k]
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
            del state[k]

    backbone.load_state_dict(state, strict=True)

    print(f"loaded {ckpt_path}")

    if args.dali:
        assert _dali_avaliable, "Dali is not currently avaiable, please install it first."
        Class = types.new_class(
            f"Dali{LinearModel.__name__}", (ClassificationABC, LinearModel))
    else:
        Class = LinearModel

    del args.backbone
    model = Class(backbone, **args.__dict__)

    train_loader, val_loader = prepare_data(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_fraction=args.data_fraction,
    )

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            save_dir=os.path.join("/media/data1/binh/solo-learn/", "linear/")
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, "linear"),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    if args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint
    else:
        ckpt_path = None

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,
        strategy=DDPPlugin(find_unused_parameters=False)
        if args.strategy == "ddp"
        else args.strategy,
    )

    if args.dali:
        model.set_loaders(val_loader=val_loader)
        trainer.fit(model, ckpt_path=ckpt_path)
    else:
        model.set_loaders(train_loader=train_loader, val_loader=val_loader)
        trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()

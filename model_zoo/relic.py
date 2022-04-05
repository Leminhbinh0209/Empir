import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
from .byol import (default, 
                    flatten,
                    singleton, 
                    get_module_device, 
                    set_requires_grad, 
                    RandomApply, EMA, 
                    RandomApply, 
                    update_moving_average,
                    get_accuracy, accuracy, get_distance
                    )

from .utils import RelicLoss



loss_fn = RelicLoss(normalize=True, temperature=1.0, alpha=0.5)
# augmentation utils
# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, layer = -2,  use_momentum=True):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.use_momentum = use_momentum
        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# main class

class ReLIC(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 128,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        use_jsd = False,
    ):
        super().__init__()
        self.net = net

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)
        self.normalize =T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]))
        self.online_encoder = NetWrapper(net, projection_size, layer=hidden_layer, use_momentum=use_momentum)
      
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

     

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)


    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True,
        target=None
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one, representation = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)
        orig_proj, _ = self.online_encoder(self.normalize(x))

        with torch.no_grad():
            target_encoder = self._get_target_encoder() 
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()
           # print(self.normalize(x))
            target_orig_img, _ = target_encoder(self.normalize(x))
            target_orig_img.detach_()

        loss_one = loss_fn(online_proj_one, target_proj_two.detach(), orig_proj)
        loss_two = loss_fn(online_proj_two, target_proj_one.detach(), orig_proj)
  
        loss = loss_one + loss_two 
        #loss = loss_one + loss_two
        if target is not None:
            top1, top5 = get_accuracy(online_proj_one.detach(), target_proj_two.detach(), target)
            return loss.mean(), top1, top5
        return loss.mean()

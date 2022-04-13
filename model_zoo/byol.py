import copy
import random
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
#from .utils import DropGrad
# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features, bias):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class BatchNorm1d_fw(nn.BatchNorm1d): #used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm1d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

class MLP_fw(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096, use_momentum=True):
        super().__init__()
        self.net = nn.Sequential(
            Linear_fw(dim, hidden_size),
            BatchNorm1d_fw(hidden_size),
            nn.ReLU(inplace=True),
            Linear_fw(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096, use_momentum=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size, bias=use_momentum),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class SimSiamProjector(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096, use_momentum=False):
        super().__init__()
        self.net =  nn.Sequential(nn.Linear(dim, hidden_size, bias=use_momentum),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(hidden_size, hidden_size, bias=use_momentum),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(hidden_size, projection_size),
                                        nn.BatchNorm1d(projection_size, affine=False)) # output layer
        self.net[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
    def forward(self, x):
        return self.net(x)
        
# Calculate accuracy
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(k*correct.shape[1]).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def get_accuracy(x1, x2, target):
    with torch.no_grad():
        output = torch.mm(x1, x2.t())
    top1 = accuracy(output, target, topk=(1, 5))
    return top1
def get_distance(x1, x2):
    x1 = F.normalize(x1, dim=-1, p=2)
    x2 = F.normalize(x2, dim=-1, p=2)

    output = torch.mm(x1, x2.t())
    return output / 0.2

# Jensen-Shannon Divergence
class JSD(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss
    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    Hacked together by / Copyright 2020 Ross Wightman
    """
    def __init__(self, num_splits=2, alpha=1.):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()


    def __call__(self, logits1, logits2):
       # print(targets.shape, logits1.shape)
        # Cross-entropy is only computed on clean images
        p_aug1, p_aug2 = F.softmax(
          logits1, dim=1), F.softmax(
              logits2, dim=1),
              
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_aug1 + p_aug2) / 2., 1e-7, 1).log()
        loss =  self.alpha * (F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 2 
        return loss
# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, 
    projection_size, 
    projection_hidden_size, 
    layer = -2,  
    use_momentum=True
    ):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
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
        if self.use_momentum:
            projector = MLP(dim, self.projection_size, self.projection_hidden_size)
            
        else:
            projector = SimSiamProjector(dim, self.projection_size, self.projection_hidden_size)
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

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        use_jsd = False,
        local_opt = False
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
        self.online_encoder = NetWrapper(net, projection_size, 
                                projection_hidden_size, 
                                layer=hidden_layer, 
                                use_momentum=use_momentum
                                )
        if use_jsd:
            self.jsd = JSD(alpha=6.)
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        # Some paprameter for optimizing online predictor
        self.local_opt = local_opt
            

        if not local_opt: # Local optimization for online predictor
            self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, use_momentum=use_momentum) # as in official SimSiam implementation
        else:
            print("Use Local Optimized Online Predictor")
            self.opt_steps = 5
            self.train_opt_lr = 0.01
            self.dropout = DropGrad('gaussian', 0.1, 'constant') # meta learning trick
            self.online_predictor = MLP_fw(projection_size, projection_size, projection_hidden_size, use_momentum=use_momentum) 
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
    def _get_parameter(self):
        predictor_param_name = []
        predictor_params = list([])
        predictor_params += list(getattr(self, "online_predictor").parameters())
        for i, param in getattr(self, "online_predictor").named_parameters():
            predictor_param_name.append(f"online_predictor.{i}")
   
        base_params = list(
            filter(lambda kv: kv[0] not in predictor_param_name, self.named_parameters()))
        base_params = [u[1] for u in base_params]

        return predictor_params, base_params
    def _optim_online_predictor(self, o1, o2, t1, t2, to):
        """        
            Optim the predictor head firt
        Args:
            o1: online_proj_one.clone().detach(),
            o2: online_proj_two, 
            t1: target_proj_one, 
            t2: target_proj_two, 
            to: target_orig_img
        return:
            None
        """
        fast_parameters = list(self.online_predictor.parameters())
        for weight in self.online_predictor.parameters():
          weight.fast = None
        self.online_predictor.zero_grad()
        for _ in range(self.opt_steps):
            op_1, op_2 =  self.online_predictor(o1), self.online_predictor(o2)

            ### Similar loss with the forward function
            loss_one = loss_fn(op_1, t2)
            loss_two = loss_fn(op_2, t1)
            jsd_regularizer = 0
            if hasattr(self, 'jsd'):
                logit1 = get_distance(op_1, to)
                logit2 = get_distance(op_2, to)
                jsd_regularizer = self.jsd(logit1, logit2)
            loss = loss_one + loss_two + jsd_regularizer

            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True) 
            grad = [g.detach() for g in grad] # first order approx
            ### Update the predictor head
            fast_parameters = []
            for k, (name, weight) in enumerate(self.online_predictor.named_parameters()):
                # regularization
                grad[k] = self.dropout(grad[k]) 
                if weight.fast is None:
                    weight.fast = weight - self.train_opt_lr * grad[k] #link fast weight to weight
                else:
                    weight.fast = weight.fast - self.train_opt_lr * grad[k]
                fast_parameters.append(weight.fast)


    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True,
        target=None
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(self.normalize(x), return_projection = return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one, representation = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)
        #print(representation.shape)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()
           # print(self.normalize(x))
            target_orig_img, _ = target_encoder(self.normalize(x))
            target_orig_img.detach_()

        if self.local_opt:
            # Foreach mini-batch, Optimize locally the online predictor. 
            # Thus, correctly predict the centor eta from one oservation.
            self._optim_online_predictor(
                online_proj_one.clone().detach(),
                online_proj_two.clone().detach(), 
                target_proj_one, 
                target_proj_two,
                target_orig_img,
            )
            online_pred_one = self.online_predictor(online_proj_one)
            online_pred_two = self.online_predictor(online_proj_two)
            

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
         
        #print(target)
        jsd_regularizer = 0
        if hasattr(self, 'jsd'):
            logit1 = get_distance(online_pred_one, target_orig_img)
            logit2 = get_distance(online_pred_two, target_orig_img)
            jsd_regularizer = self.jsd(logit1, logit2) 

        loss = loss_one + loss_two + jsd_regularizer
        #loss = loss_one + loss_two
        if target is not None:
            top1, top5 = get_accuracy(online_pred_one.detach(), target_proj_two.detach(), target)
            return loss.mean(), top1, top5
        return loss.mean()

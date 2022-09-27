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

import argparse
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def sync_tensor_across_gpus(t: Union[torch.Tensor, None]
                            ) -> Union[torch.Tensor, None]:
    # t needs to have dim 0 for troch.cat below.
    # if not, you need to prepare it.
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    # this works with nccl backend when tensors need to be on gpu.
    dist.all_gather(gather_t_tensor, t)
   # for gloo and mpi backends, tensors need to be on cpu. also this works single machine with
   # multiple   gpus. for multiple nodes, you should use dist.all_gather_multigpu. both have the
   # same definition... see [here](https://pytorch.org/docs/stable/distributed.html).
   #  somewhere in the same page, it was mentioned that dist.all_gather_multigpu is more for
   # multi-nodes. still dont see the benefit of all_gather_multigpu. the provided working case in
   # the doc is  vague...
    return torch.cat(gather_t_tensor, dim=0)


def get_distance(x1, x2):
    x1 = F.normalize(x1, dim=-1, p=2)
    x2 = F.normalize(x2, dim=-1, p=2)

    output = torch.mm(x1, x2.t())
    return output / 0.2


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
        loss = self.alpha * (F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                             F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 2
        return loss


class JSD3(nn.Module):
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

    def __call__(self, logits1, logits2, original):
        # print(targets.shape, logits1.shape)
        # Cross-entropy is only computed on clean images
        p_aug1, p_aug2 = F.softmax(
            logits1, dim=1), F.softmax(
            logits2, dim=1),
        ori_aug = F.softmax(original, dim=1)
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp(
            (p_aug1 + p_aug2 + ori_aug) / 3., 1e-7, 1).log()
        loss = self.alpha * (F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                             F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 2
        return loss


class BYOL(BaseMomentumMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        **kwargs,
    ):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(**kwargs)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )
        self.apply_jsd = kwargs.get("apply_jsd") == True
        self.strategy = kwargs.get("strategy")
        if self.apply_jsd:
            assert kwargs["with_origin_image"], "No original Image for momentum"
            self.alpha_jsd = kwargs.get("alpha_jsd")
            self.jsd_loss = JSD(alpha=self.alpha_jsd)
            self.topk = kwargs.get("topk")
            print(f"Apply JSD with topk = {self.topk}- w={self.alpha_jsd}")

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(
            BYOL, BYOL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("byol")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of
                the parent and the momentum projected features.
        """

        out = super().momentum_forward(X)
        z = self.momentum_projector(out["feats"])
        out.update({"z": z})
        return out

    def _shared_step(
        self, feats: List[torch.Tensor], momentum_feats: List[torch.Tensor]
    ) -> torch.Tensor:

        Z = [self.projector(f) for f in feats]
        P = [self.predictor(z) for z in Z]

        # forward momentum backbone
        with torch.no_grad():
            Z_momentum = [self.momentum_projector(f) for f in momentum_feats]

        # ------- negative consine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(
                Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        return neg_cos_sim, z_std

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        Z = out["z"]
        P = out["p"]
        Z_momentum = out["momentum_z"]

        # ------- negative consine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])
        jsd_reg = 0
        if self.apply_jsd:
            top_k = self.topk
            p1 = P[0]
            p2 = P[1]
            # original image Z_momentum[2]
            origin_embedding = Z_momentum[self.num_crops]

            total_p1 = torch.cat(FullGatherLayer.apply(
                p1), dim=0) if self.strategy == "ddp" else p1
            # total_p1 = sync_tensor_across_gpus(p1).clone().detach()
            # total_p1[local_rank * batch_size: (local_rank+1)*batch_size] = p1

            total_p2 = torch.cat(FullGatherLayer.apply(
                p2), dim=0) if self.strategy == "ddp" else p2
            # total_p2 = sync_tensor_across_gpus(p2).clone().detach()
            # total_p2[local_rank * batch_size: (local_rank+1)*batch_size] = p2

            # sync_tensor_across_gpus(origin_embedding)
            #
            total_zo = torch.cat(
                FullGatherLayer.apply(origin_embedding), dim=0) if self.strategy == "ddp" else origin_embedding

            # jsd_reg = self.jsd_loss(total_p1, total_p2, total_zo)

            logit1 = get_distance(total_p1, total_zo)  # batchsize x batch_size
            logit2 = get_distance(total_p2, total_zo)
            a_arg = torch.argsort(logit1, dim=1, descending=True)
            b_arg = torch.argsort(logit2, dim=1, descending=True)

            logit1_a_sort = torch.gather(
                input=logit1, dim=1, index=a_arg)[:, :top_k]
            logit2_a_arg = torch.gather(
                input=logit2, dim=1, index=a_arg)[:, :top_k]

            logit1_b_sort = torch.gather(
                input=logit1, dim=1, index=b_arg)[:, :top_k]
            logit2_b_arg = torch.gather(
                input=logit2, dim=1, index=b_arg)[:, :top_k]

            jsd_reg = 0.5 * self.jsd_loss(logit1_a_sort, logit2_a_arg) +\
                0.5 * self.jsd_loss(logit1_b_sort, logit2_b_arg)

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(
                Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "train_jsd": jsd_reg
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss + jsd_reg

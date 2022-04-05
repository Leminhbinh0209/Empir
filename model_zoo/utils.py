

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class RelicLoss(nn.Module):
    
    def __init__(self, normalize=True, temperature=1.0, alpha=0.5):
        super(RelicLoss, self).__init__()
        self.normalize = normalize
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, zi, zj, z_orig):
        bs = zi.shape[0]
        labels = torch.zeros((2*bs,)).long().to(zi.device)
        mask = torch.ones((bs, bs), dtype=bool).fill_diagonal_(0)

        if self.normalize:
            zi_norm = F.normalize(zi, p=2, dim=-1)
            zj_norm = F.normalize(zj, p=2, dim=-1)
            zo_norm = F.normalize(z_orig, p=2, dim=-1)
        else:
            zi_norm = zi
            zj_norm = zj
            zo_norm = z_orig

        logits_ii = torch.mm(zi_norm, zi_norm.t()) / self.temperature
        logits_ij = torch.mm(zi_norm, zj_norm.t()) / self.temperature
        logits_ji = torch.mm(zj_norm, zi_norm.t()) / self.temperature
        logits_jj = torch.mm(zj_norm, zj_norm.t()) / self.temperature

        logits_ij_pos = logits_ij[torch.logical_not(mask)]                                          # Shape (N,)
        logits_ji_pos = logits_ji[torch.logical_not(mask)]                                          # Shape (N,)
        logits_ii_neg = logits_ii[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_ij_neg = logits_ij[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_ji_neg = logits_ji[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_jj_neg = logits_jj[mask].reshape(bs, -1)                                             # Shape (N, N-1)

        pos = torch.cat((logits_ij_pos, logits_ji_pos), dim=0).unsqueeze(1)                         # Shape (2N, 1)
        neg_i = torch.cat((logits_ii_neg, logits_ij_neg), dim=1)                                    # Shape (N, 2N-2)
        neg_j = torch.cat((logits_ji_neg, logits_jj_neg), dim=1)                                    # Shape (N, 2N-2)
        neg = torch.cat((neg_i, neg_j), dim=0)                                                      # Shape (2N, 2N-2)

        logits = torch.cat((pos, neg), dim=1)                                                       # Shape (2N, 2N-1)
        contrastive_loss = F.cross_entropy(logits, labels)

        logits_io = torch.mm(zi_norm, zo_norm.t()) / self.temperature
        logits_jo = torch.mm(zj_norm, zo_norm.t()) / self.temperature
        probs_io = F.softmax(logits_io[torch.logical_not(mask)], -1)
        probs_jo = F.log_softmax(logits_jo[torch.logical_not(mask)], -1)
        kl_div_loss = F.kl_div(probs_io, probs_jo, log_target=True, reduction="sum")
        return contrastive_loss + self.alpha * kl_div_loss

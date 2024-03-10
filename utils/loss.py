import torch
import torch.nn as nn
# Rank Loss (Pairwise Ranking Loss)
class RankLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(RankLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_pred, neg_pred):
        # The loss is max(0, margin - pos_pred + neg_pred)
        loss = torch.clamp(self.margin - pos_pred + neg_pred, min=0)
        return loss.mean()


class TolerantMSELossv1(nn.Module):
    def __init__(self, margin=1.0):
        super(TolerantMSELossv1, self).__init__()
        self.margin = margin

    def forward(self,pred, true):
        loss = (self.margin-torch.abs(pred-true))**2 
        return loss.mean()


class TolerantMSELossv2(nn.Module):
    def __init__(self, margin=1.0):
        super(TolerantMSELossv2, self).__init__()
        self.margin = margin

    def forward(self, pred, true):
        # The loss is 
        n = torch.abs(pred - true)
        
        loss = torch.clamp(n - self.margin, min=0)
        return loss.mean()

class CVAELoss(nn.Module):
    def __init__(self):
        super(CVAELoss, self).__init__()
    def forward(input_ids,reconstructed_logits, mu, log_var):
        # Reconstruction loss
        reconstruction_loss = nn.CrossEntropyLoss()(reconstructed_logits.view(-1, reconstructed_logits.size(-1)), input_ids.view(-1))

        # KL divergence
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return reconstruction_loss, kl_divergence
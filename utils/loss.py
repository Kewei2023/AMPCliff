import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List


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


class OrthogonalLoss(nn.Module):
    """
    Wrapper that combines task loss with orthogonal constraint loss.

    This module wraps a task loss function and adds orthogonal regularization
    on layer representations.

    Args:
        task_loss_fn: The primary task loss function (e.g., nn.MSELoss())
        orthogonal_weight: Weight for the orthogonal constraint term (default: 0.01)
        layer_indices: Which layers to constrain. None = all layers.
        constraint_type: "pairwise", "sequential", or "gram"
        normalize: Whether to L2 normalize before computing orthogonality
        log_metrics: Whether to return detailed metrics dictionary

    Example:
        >>> loss_fn = OrthogonalLoss(
        ...     task_loss_fn=nn.MSELoss(),
        ...     orthogonal_weight=0.01
        ... )
        >>> predictions = model(inputs)
        >>> loss, metrics = loss_fn(predictions, targets, layer_reps)
    """

    def __init__(
        self,
        task_loss_fn: nn.Module,
        orthogonal_weight: float = 0.01,
        layer_indices: Optional[List[int]] = None,
        constraint_type: str = "pairwise",
        normalize: bool = True,
        log_metrics: bool = True
    ):
        super().__init__()
        from .orthogonal_constraint import OrthogonalConstraint

        self.task_loss_fn = task_loss_fn
        self.orthogonal_weight = orthogonal_weight
        self.orthogonal_constraint = OrthogonalConstraint(
            layer_indices=layer_indices,
            constraint_type=constraint_type,
            normalize=normalize
        )
        self.log_metrics = log_metrics

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        layer_representations: Union[torch.Tensor, List[torch.Tensor]]
    ) -> tuple:
        """
        Compute combined task loss and orthogonal constraint loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            layer_representations: Layer outputs [B, D, L] or List[Tensor]

        Returns:
            total_loss: Combined loss (task + orthogonal)
            loss_dict: Dictionary with individual loss components
        """
        # Compute task loss
        task_loss = self.task_loss_fn(predictions, targets)

        # Compute orthogonal constraint loss
        ortho_loss = self.orthogonal_constraint(layer_representations)

        # Combined loss
        total_loss = task_loss + self.orthogonal_weight * ortho_loss

        # Build metrics dictionary
        loss_dict = {
            'task_loss': task_loss,
            'ortho_loss': ortho_loss,
            'total_loss': total_loss
        }

        return total_loss, loss_dict
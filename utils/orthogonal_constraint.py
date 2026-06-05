"""
Orthogonal Constraint Module for AMPCliff.

This module implements orthogonality regularization between layer representations
to encourage diverse and non-redundant feature learning across layers.

Core formula:
    L_orth = sum_{i<j} |<h_i, h_j>|^2 / (||h_i||^2 * ||h_j||^2)

where h_i is the pooled representation from layer i.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class OrthogonalConstraint(nn.Module):
    """
    Computes orthogonality regularization loss between layer representations.

    This constraint encourages different layers to learn diverse, non-redundant
    representations by penalizing similarity between layer outputs.

    Args:
        layer_indices: Which layers to constrain. None means all layers.
                      Example: [0, 2, 4] constrains only layers 0, 2, and 4.
        constraint_type: Type of constraint:
            - "pairwise": All pairwise layer combinations (default)
            - "sequential": Only adjacent layer pairs
            - "gram": Gram matrix based constraint
        normalize: Whether to L2 normalize representations before computing
                  orthogonality. Recommended for stability.
        reduction: How to aggregate the loss:
            - "mean": Average over all pairs (default)
            - "sum": Sum over all pairs
            - "none": Return full orthogonality matrix
        eps: Small constant for numerical stability in normalization

    Example:
        >>> constraint = OrthogonalConstraint(
        ...     layer_indices=[0, 2, 4, 6],
        ...     constraint_type="pairwise",
        ...     normalize=True
        ... )
        >>> layer_reps = torch.randn(4, 128, 6)  # [B, D, L]
        >>> loss = constraint(layer_reps)
    """

    def __init__(
        self,
        layer_indices: Optional[List[int]] = None,
        constraint_type: str = "pairwise",
        normalize: bool = True,
        reduction: str = "mean",
        eps: float = 1e-8
    ):
        super().__init__()

        # Validate constraint type
        valid_types = ["pairwise", "sequential", "gram"]
        if constraint_type not in valid_types:
            raise ValueError(
                f"constraint_type must be one of {valid_types}, got '{constraint_type}'"
            )

        # Validate reduction
        valid_reductions = ["mean", "sum", "none"]
        if reduction not in valid_reductions:
            raise ValueError(
                f"reduction must be one of {valid_reductions}, got '{reduction}'"
            )

        self.layer_indices = layer_indices
        self.constraint_type = constraint_type
        self.normalize = normalize
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        layer_representations: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute orthogonality loss from layer representations.

        Args:
            layer_representations: Layer outputs in one of two formats:
                - torch.Tensor of shape [B, D, L] where B=batch, D=hidden_dim, L=num_layers
                - List[Tensor] where each tensor is [B, D] for each layer

        Returns:
            Orthogonality loss scalar (or matrix if reduction="none")
        """
        # Convert list to tensor if needed
        if isinstance(layer_representations, list):
            if len(layer_representations) == 0:
                raise ValueError("Empty layer representations list")
            layer_representations = torch.stack(layer_representations, dim=-1)

        # layer_representations: [B, D, L]
        if layer_representations.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [B, D, L], got shape {layer_representations.shape}"
            )

        batch_size, hidden_dim, num_layers = layer_representations.shape

        # Select specific layers if layer_indices is specified
        if self.layer_indices is not None:
            if len(self.layer_indices) == 0:
                raise ValueError("layer_indices cannot be empty")
            if max(self.layer_indices) >= num_layers:
                raise IndexError(
                    f"layer_indices {self.layer_indices} out of range for {num_layers} layers"
                )
            layer_representations = layer_representations[:, :, self.layer_indices]
            num_layers = len(self.layer_indices)

        # Single layer: no pairs to compare
        if num_layers < 2:
            if self.reduction == "none":
                return torch.zeros(num_layers, num_layers, device=layer_representations.device)
            return torch.tensor(0.0, device=layer_representations.device)

        # Compute orthogonality matrix
        ortho_matrix = self.compute_orthogonality_matrix(layer_representations)

        # Compute loss based on constraint type
        if self.constraint_type == "pairwise":
            loss = self._compute_pairwise_loss(ortho_matrix)
        elif self.constraint_type == "sequential":
            loss = self._compute_sequential_loss(ortho_matrix)
        elif self.constraint_type == "gram":
            loss = self._compute_gram_loss(ortho_matrix)
        else:
            raise ValueError(f"Unknown constraint_type: {self.constraint_type}")

        return loss

    def compute_orthogonality_matrix(
        self,
        layer_representations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L x L orthogonality matrix showing similarity between layers.

        Args:
            layer_representations: [B, D, L] tensor

        Returns:
            ortho_matrix: [L, L] matrix where entry (i,j) is the average
                         cosine similarity between layers i and j across batches
        """
        # layer_representations: [B, D, L]
        B, D, L = layer_representations.shape

        # Reshape to [B, L, D] for easier computation
        layer_reps = layer_representations.permute(0, 2, 1)  # [B, L, D]

        if self.normalize:
            # L2 normalize each layer representation
            norms = torch.norm(layer_reps, p=2, dim=-1, keepdim=True)  # [B, L, 1]
            norms = norms.clamp(min=self.eps)  # Avoid division by zero
            layer_reps = layer_reps / norms

        # Compute pairwise cosine similarities: [B, L, L]
        # cos_sim(i,j) = (h_i . h_j) / (||h_i|| * ||h_j||)
        # Since we normalized, it's just dot product
        ortho_matrix = torch.bmm(layer_reps, layer_reps.transpose(1, 2))  # [B, L, L]

        # Average across batch dimension
        ortho_matrix = ortho_matrix.mean(dim=0)  # [L, L]

        return ortho_matrix

    def _compute_pairwise_loss(self, ortho_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for all pairwise layer combinations.

        Loss = mean/sum of |cos_sim(i,j)|^2 for all i < j
        """
        L = ortho_matrix.shape[0]

        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(L, L, device=ortho_matrix.device), diagonal=1).bool()
        upper_tri = ortho_matrix[mask]

        # Loss is squared cosine similarity (we want orthogonal = 0 similarity)
        pair_losses = upper_tri ** 2

        if self.reduction == "none":
            # Return full matrix of losses
            loss_matrix = torch.zeros_like(ortho_matrix)
            loss_matrix[mask] = pair_losses
            loss_matrix = loss_matrix + loss_matrix.T  # Make symmetric
            return loss_matrix
        elif self.reduction == "sum":
            return pair_losses.sum()
        else:  # mean
            num_pairs = L * (L - 1) // 2
            if num_pairs == 0:
                return torch.tensor(0.0, device=ortho_matrix.device)
            return pair_losses.sum() / num_pairs

    def _compute_sequential_loss(self, ortho_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for only adjacent layer pairs.

        Loss = mean/sum of |cos_sim(i, i+1)|^2 for i in [0, L-1)
        """
        L = ortho_matrix.shape[0]

        # Get diagonal + 1 (adjacent pairs)
        adjacent_losses = []
        for i in range(L - 1):
            adjacent_losses.append(ortho_matrix[i, i + 1] ** 2)

        if len(adjacent_losses) == 0:
            return torch.tensor(0.0, device=ortho_matrix.device)

        adjacent_losses = torch.stack(adjacent_losses)

        if self.reduction == "none":
            # Return matrix with only adjacent pairs
            loss_matrix = torch.zeros_like(ortho_matrix)
            for i in range(L - 1):
                loss_matrix[i, i + 1] = adjacent_losses[i]
                loss_matrix[i + 1, i] = adjacent_losses[i]
            return loss_matrix
        elif self.reduction == "sum":
            return adjacent_losses.sum()
        else:  # mean
            return adjacent_losses.mean()

    def _compute_gram_loss(self, ortho_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix based orthogonality loss.

        Loss = ||G - I||_F^2 / L
        where G is the Gram matrix (ortho_matrix) and I is identity.

        This encourages the Gram matrix to be identity (orthonormal).
        """
        L = ortho_matrix.shape[0]

        # Identity matrix
        identity = torch.eye(L, device=ortho_matrix.device)

        # Frobenius norm squared of difference
        loss = torch.norm(ortho_matrix - identity, p='fro') ** 2

        if self.reduction == "none":
            return (ortho_matrix - identity) ** 2
        elif self.reduction == "sum":
            return loss
        else:  # mean
            return loss / (L * L)

    def extra_repr(self) -> str:
        """Return string representation for print()."""
        return (
            f"layer_indices={self.layer_indices}, "
            f"constraint_type='{self.constraint_type}', "
            f"normalize={self.normalize}, "
            f"reduction='{self.reduction}'"
        )

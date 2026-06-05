import torch
import torch.nn as nn

# import contextlib

# @contextlib.contextmanager
# def eval_no_grad(model: nn.Module):
#     was_training = model.training
#     model.eval()
#     with torch.no_grad():
#         yield
#     if was_training:
        # model.train()

def knock_rows_cols_in_probs(probs: torch.Tensor,
                             row_indices=None, col_indices=None,
                             renorm: bool = False):
    """
    probs: [B, H, T, T] 注意力 softmax 后概率
    row_indices: 置零哪些行（这些 query 不再关注任何 key）
    col_indices: 置零哪些列（这些 key 不再被任何 query 关注）
    renorm: 是否对每一行重归一化使其和为1（行全零时保持全零）
    """
    if row_indices is not None and len(row_indices) > 0:
        probs[..., row_indices, :] = 0.0
    if col_indices is not None and len(col_indices) > 0:
        probs[..., :, col_indices] = 0.0

    if renorm:
        row_sums = probs.sum(dim=-1, keepdim=True)   # [B,H,T,1]
        nonzero = row_sums > 0
        probs = torch.where(
            nonzero,
            probs / (row_sums + 1e-12),
            probs  # 行全零时保持全零（等效于不输出任何 value）
        )
    return probs




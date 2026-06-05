import torch
# from utils import knock_rows_cols_in_probs

class HiddenStateMaskHook:
    """
    在 encoder.layer[L] 的 forward_pre_hook 上，将输入 hidden_states 的指定 token 置零。
    - layer_ids: 需要注入 KO 的层索引列表（如 [0,1,...,L-1] 或 [L*]）
    - token_positions: 每个 batch 序列要 KO 的 token 索引（统一整数位置；需自行保证未越界与非 pad）
    - mask_qkv: ('q','k','v','all') 仅对 self-attention 里某分支置零时，可选更细粒度；默认 'all' 直接置零隐状态（影响 Q/K/V）
    """
    def __init__(self, encoder, layer_ids, token_positions, mask_qkv='all'):
        self.encoder = encoder
        self.layer_ids = set(int(i) for i in layer_ids)
        self.token_positions = token_positions  # list[int] or torch.LongTensor shape [B]
        self.mask_qkv = mask_qkv
        self.handles = []

    def _pre_hook(self, layer_id: int):
        def fn(module, inputs):
            # inputs: tuple(hidden_states, attention_mask, ...)
            hidden_states = inputs[0]  # [B, T, D]
            B, T, D = hidden_states.shape
            pos = self.token_positions
            if isinstance(pos, torch.Tensor):
                assert pos.shape[0] == B, "token_positions per-batch required"
                pos_ = pos
            else:
                pos_ = torch.tensor(pos, device=hidden_states.device).view(1).repeat(B)
            # 直接置零该 token 的隐状态（影响 Q/K/V）
            hs = hidden_states.clone()
            hs[torch.arange(B, device=hs.device), pos_, :] = 0.0
            # 返回新的 inputs 元组
            new_inputs = (hs,) + inputs[1:]
            return new_inputs
        return fn

    def register(self):
        layers = self.encoder.layer
        for lid, layer in enumerate(layers):
            if lid in self.layer_ids:
                h = layer.register_forward_pre_hook(self._pre_hook(lid))
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

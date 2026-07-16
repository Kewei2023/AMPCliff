# maintained by kewei li
import types
import inspect
import torch
import torch.nn as nn

class _NoOpRotaryEmbedding(nn.Module):
    """用于新式接口：forward(q, k, seq_len=None, position_ids=None, **kw) -> (q, k)"""
    def forward(self, q, k, *args, **kwargs):
        # print("**********************")
        # print("*no rotary embeddings*")
        # print("**********************")
        return q, k


class DisableRoPEHookHF:
    """
    在指定层禁用 RoPE（将旋转变为恒等映射）。
    - 优先尝试“新式接口”替换 attn.rotary_emb 为 _NoOpRotaryEmbedding。
    - 若检测为“旧式接口”（rotary_emb 返回 cos/sin），则替换为 _FakeRotaryOldAPI（cos=1, sin=0）。
    该 hook 仅改写模块属性，不 monkey-patch 全局函数，易于恢复。
    """
    def __init__(self, hf_model_encoder: nn.Module, target_layers):
        self.hf_model_encoder = hf_model_encoder
        self.target_layers = list(target_layers)
        self._records = []  # 保存 (attn_module, old_rotary_emb)

    

    def register(self):
        layers = self.hf_model_encoder.layer
        for L in self.target_layers:
            layer = layers[L]
            attn = layer.attention.self
            if not hasattr(attn, "rotary_embeddings"):
                # 某些实现把旋转逻辑写进 forward；若遇到这种模型，需要定制化 monkey-patch forward。
                raise AttributeError(f"Layer {L} 的注意力模块不含 rotary_embeddings")

            old = attn.rotary_embeddings
            attn.rotary_embeddings = _NoOpRotaryEmbedding()
            self._records.append((attn, old))

    def remove(self):
        for attn, old in self._records:
            attn.rotary_embeddings = old
        self._records.clear()

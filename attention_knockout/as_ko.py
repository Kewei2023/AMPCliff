# maintained by kewei li
# post_softmax_knockout.py
import types
import math
import torch
import torch.nn.functional as F
from .utils import knock_rows_cols_in_probs
import functools
import ipdb

class PostSoftmaxAttentionKnockout:
    def __init__(self, encoder, layer_ids, row_indices=None, col_indices=None, renorm=True):
        self.encoder = encoder
        self.layer_ids = set(int(i) for i in layer_ids)
        self.row_indices = row_indices if row_indices is not None else []
        self.col_indices = col_indices if col_indices is not None else []
        self.renorm = renorm
        self._orig_forwards = {}

    def _wrap_forward(self, attn_module, layer_id: int):
        # 保留未绑定函数，避免二次绑定导致参数错位
        orig_fwd_unbound = attn_module.__class__.forward

        def patched_forward(module_self, *args, **kwargs):
            kwargs_mod = dict(kwargs)

            # —— 不再强制塞 output_attentions —— 
            # 若你仍想兜底，可做“谨慎补”：只有当原函数签名包含该形参、且
            #   1) 调用方没以关键字给出，且
            #   2) 也没以位置参数给出（避免重复）
            # 时才补 True。
            try:
                sig = inspect.signature(orig_fwd_unbound)
                if 'output_attentions' in sig.parameters:
                    # 找到该参数的位置索引（用于判断是否已由位置参数提供）
                    pos = list(sig.parameters).index('output_attentions')
                    if ('output_attentions' not in kwargs_mod) and (len(args) <= pos):
                        kwargs_mod['output_attentions'] = True
            except Exception:
                # 安静失败：签名取不到就不补
                pass

            out = orig_fwd_unbound(module_self, *args, **kwargs_mod)

            # 期望返回：(attn_output, attn_probs, ...)
            if not (isinstance(out, tuple) and len(out) >= 2 and torch.is_tensor(out[1])):
                return out

            attn_output, attn_probs = out[0], out[1]  # attn_probs: [B, H, T, T]

            # 在概率上做 KO（post-softmax）
            probs_mod = knock_rows_cols_in_probs(attn_probs,
                                                 self.row_indices,
                                                 self.col_indices,
                                                 self.renorm)

            # 近似法：仅在 renorm=False 时，用行和缩放输出（避免尺度飘移）
            if not self.renorm:
                rowmass = probs_mod.sum(dim=-1, keepdim=False)   # [B, H, T]
                scale = rowmass.mean(dim=1, keepdim=True)        # [B, 1, T]
                scale = scale.transpose(-1, -2)                  # [B, T, 1]
                attn_output = attn_output * scale

            out_list = list(out)
            out_list[0] = attn_output
            out_list[1] = probs_mod
            return tuple(out_list)

        return types.MethodType(patched_forward, attn_module)

    def register(self):
        for lid, layer in enumerate(self.encoder.layer):
            if lid in self.layer_ids:
                attn = getattr(layer, "attention", None) or getattr(layer, "self_attn", None) or getattr(layer, "self", None)
                if attn is None:
                    raise RuntimeError(f"layer {lid} has no attention module attribute to wrap")
                if id(attn) in self._orig_forwards:
                    continue
                self._orig_forwards[id(attn)] = attn.forward
                attn.forward = self._wrap_forward(attn, lid)

    def remove(self):
        for lid, layer in enumerate(self.encoder.layer):
            attn = getattr(layer, "attention", None) or getattr(layer, "self_attn", None) or getattr(layer, "self", None)
            if attn is not None and id(attn) in self._orig_forwards:
                attn.forward = self._orig_forwards[id(attn)]
        self._orig_forwards.clear()


class PreSoftmaxAttentionKnockout:
    def __init__(self, encoder, layer_ids, # dtype,device,
                     row_indices=None,
                      col_indices=None
                      ):
        self.encoder = encoder
        # self.dtype = dtype
        # self.device = device
        self.layer_ids = set(int(i) for i in layer_ids)
        self.row_indices = row_indices if row_indices is not None else []
        # self.col_indices = col_indices if col_indices is not None else []
        # self.from_to_index_ = [tuple(row_indices),tuple(col_indices)]
        # self.renorm = renorm
        self._orig_forwards_hook = []

    def _wrap_forward(self, attn_forward,opposite_):
        @functools.wraps(attn_forward)
        def patched_forward(*args, **kwargs):
            
            '''
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            '''
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for (k, v) in kwargs.items():
                new_kwargs[k] = v
            # ipdb.set_trace()
            num_tokens = new_args[0][0].shape[0]
            attn_mask = new_args[1]
            seq_len = (new_args[1]==0).sum().item()
            dtype = attn_mask.dtype
            device = attn_mask.device
            # ipdb.set_trace()
            # 1 visable 0 mask
            if opposite_:
                # all mask, visable the index
                attn_mask_expand = torch.zeros_like(attn_mask)
                
                if len(self.row_indices) == 1:
                    # if len(self.col_indices) > 1:
                    #     attn_mask_expand[...,self.row_indices[0]] = 1
                    # else:
                    attn_mask_expand[...,self.row_indices[0]] = 1
                # elif len(self.col_indices) > 1:
                #     attn_mask_expand[self.row_indices[0]:self.row_indices[1], self.col_indices[0]:self.col_indices[-1]] = 1
                else:
                    attn_mask_expand[...,self.row_indices[0]:self.row_indices[1]] = 1
                attn_mask_expand = attn_mask_expand.to(dtype=dtype)  # fp16 compatibility
                attn_mask_expand = (1.0 - attn_mask_expand) * torch.finfo(dtype).min
            else:
                # all visable, mask the index
                attn_mask_expand = attn_mask.clone()
                # attn_mask_expand[...,seq_len:] = 0
                if len(self.row_indices) == 1:
                    # if len(self.col_indices) > 1:
                    #     attn_mask_expand[self.row_indices[0], self.col_indices[0]:self.col_indices[-1]] = 0
                    # else:
                    attn_mask_expand[...,self.row_indices[0]] = torch.finfo(dtype).min
                # elif len(self.col_indices) > 1:
                #     attn_mask_expand[self.row_indices[0]:self.row_indices[1], self.col_indices[0]:self.col_indices[-1]] = 0
                else:
                    attn_mask_expand[...,self.row_indices[0]:self.row_indices[1]] = torch.finfo(dtype).min
            
            # attn_mask_new = attn_mask_expand.unsqueeze(0).unsqueeze(0)
            
            # attn_mask_new = attn_mask_new.to(device)
            # ipdb.set_trace()
            new_args[1] = attn_mask_expand
            return attn_forward(*new_args, **new_kwargs)
            

        return patched_forward

    def register(self,opposite=False):
        for lid, layer in enumerate(self.encoder.layer):
            if lid in self.layer_ids:
                attn = layer.attention.self
                if attn is None:
                    raise RuntimeError(f"layer {lid} has no attention module attribute to wrap")
                
                self._orig_forwards_hook.append((lid,attn.forward))
                # ipdb.set_trace()
                attn.forward = self._wrap_forward(attn.forward,opposite)

    def remove(self):
        for i, hook in self._orig_forwards_hook:
            self.encoder.layer[i].attention.self.forward =hook
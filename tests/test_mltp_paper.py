from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = REPO_ROOT.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mltp_mod = _load_module(
    "AMPCliff.factory.pooling.MultiLayersTrainablePooling",
    "factory/pooling/MultiLayersTrainablePooling.py",
)
_llm = _load_module(
    "AMPCliff.factory.pooling.llm_pooling_dropin_test",
    "factory/pooling/llm_pooling_dropin.py",
)

OfficialMLTPPooling = _llm.OfficialMLTPPooling
PerceiverResampler = _mltp_mod.PerceiverResampler
resolve_mltp_kwargs = _llm.resolve_mltp_kwargs
per_layer_masked_mean = _llm.per_layer_masked_mean


def _config_for_version(version: str):
    presets = {
        "esm2_t6": (320, 6),
        "esm2_t12": (480, 12),
    }
    hidden_size, num_layers = presets[version]
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        version=version,
    )


def _make_pooler(version: str = "esm2_t6", normalize_output: bool = True):
    config = _config_for_version(version)
    return OfficialMLTPPooling(
        version=version,
        config=config,
        pooling_kwargs={"normalize_output": normalize_output},
    )


def test_mltp_paper_output_shape():
    torch.manual_seed(0)
    B, T, D, L = 2, 11, 320, 6
    all_hidden = torch.randn(B, T, D, L)
    mask = torch.ones(B, T)
    mask[:, -2:] = 0

    pooler = _make_pooler()
    out = pooler(all_hidden, mask)
    assert out.shape == (B, D)
    assert torch.isfinite(out).all()


def test_mltp_paper_gradients_nonzero():
    torch.manual_seed(0)
    B, T, D, L = 2, 11, 320, 6
    all_hidden = torch.randn(B, T, D, L, requires_grad=True)
    mask = torch.ones(B, T)

    pooler = _make_pooler()
    out = pooler(all_hidden, mask)
    out.sum().backward()

    assert pooler.resampler.latents.grad is not None
    assert pooler.resampler.latents.grad.norm().item() > 0
    assert pooler.resampler.pos_emb.weight.grad is not None
    assert pooler.resampler.pos_emb.weight.grad.norm().item() > 0

    cross_attn_grad = sum(
        p.grad.norm().item()
        for p in pooler.resampler.cross_attend_blocks[0].parameters()
        if p.grad is not None
    )
    assert cross_attn_grad > 0


def test_mltp_paper_padding_excluded_from_per_layer_mean():
    torch.manual_seed(0)
    B, T, D, L = 1, 5, 8, 2
    all_hidden = torch.zeros(B, T, D, L)
    all_hidden[:, 0, :, :] = 1.0
    all_hidden[:, 1, :, :] = 3.0
    all_hidden[:, 2:, :, :] = 999.0

    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
    layer_repr = per_layer_masked_mean(all_hidden, mask)
    assert torch.allclose(layer_repr[:, :, 0], torch.tensor(2.0))
    assert (layer_repr != 999.0).all()


def test_mltp_paper_output_changes_when_one_layer_changes():
    torch.manual_seed(0)
    B, T, D, L = 2, 8, 320, 6
    all_hidden = torch.randn(B, T, D, L)
    mask = torch.ones(B, T)
    pooler = _make_pooler()

    out1 = pooler(all_hidden, mask)
    modified = all_hidden.clone()
    modified[:, :, :, 2] = 100.0
    out2 = pooler(modified, mask)
    assert (out1 - out2).abs().max().item() > 1e-3


def test_mltp_paper_layer_count_mismatch_raises():
    pooler = _make_pooler()
    all_hidden = torch.randn(1, 5, 320, 3)
    with pytest.raises(ValueError, match="expected 6 hidden layers"):
        pooler(all_hidden, torch.ones(1, 5))


def test_mltp_paper_normalize_output():
    torch.manual_seed(0)
    B, T, D, L = 2, 7, 320, 6
    pooler = _make_pooler(normalize_output=True)
    out = pooler(torch.randn(B, T, D, L), torch.ones(B, T))
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)


def test_mltp_paper_uses_perceiver_resampler():
    pooler = _make_pooler()
    assert isinstance(pooler.resampler, PerceiverResampler)


def test_mltp_preset_esm2_t12():
    torch.manual_seed(0)
    pooler = _make_pooler(version="esm2_t12")
    B, T, D, L = 2, 9, 480, 12
    out = pooler(torch.randn(B, T, D, L), torch.ones(B, T))
    assert out.shape == (B, D)


def test_mltp_preset_mismatch_raises():
    config = SimpleNamespace(hidden_size=320, num_hidden_layers=12)
    with pytest.raises(ValueError, match="expects num_hidden_layers=6"):
        resolve_mltp_kwargs("esm2_t6", config, {})


def test_mltp_unknown_version_raises():
    config = _config_for_version("esm2_t6")
    with pytest.raises(ValueError, match="Unknown MLTP preset"):
        resolve_mltp_kwargs("esm2_t33", config, {})


def test_mltp_yaml_override_num_latents():
    config = _config_for_version("esm2_t6")
    resolved = resolve_mltp_kwargs("esm2_t6", config, {"num_latents": 48})
    assert resolved["num_latents"] == 48


def test_mltp_preset_uses_60_latents_not_pooling_common():
    config = _config_for_version("esm2_t6")
    pooler = OfficialMLTPPooling("esm2_t6", config, pooling_kwargs={"num_latents": 8})
    # Explicit override still works
    assert pooler.resampler.latents.shape[0] == 8

    pooler_default = OfficialMLTPPooling("esm2_t6", config, pooling_kwargs={})
    assert pooler_default.resampler.latents.shape[0] == 60

from types import SimpleNamespace

import pytest
import torch

from AMPCliff.factory.pooling import get_supported_poolings, validate_pooling_name
from AMPCliff.factory.pooling.registry import build_pooling_modules
from AMPCliff.factory.regression import ClassificationHead2


RELEASE_POOLINGS = (
    "mean",
    "max",
    "last",
    "latent_attn",
    "attn_structured",
    "mltp_paper",
    "FLaG",
)


def test_supported_poolings_match_release_list():
    assert get_supported_poolings() == RELEASE_POOLINGS


@pytest.mark.parametrize("pooling", RELEASE_POOLINGS)
def test_validate_pooling_name_accepts_release_poolings(pooling):
    assert validate_pooling_name(pooling) == pooling


@pytest.mark.parametrize(
    "pooling",
    ["mean", "max", "last", "latent_attn", "attn_structured", "FLaG"],
)
def test_regression_head2_token_poolings(pooling):
    cfg = SimpleNamespace(
        hidden_size=64,
        hidden_dropout_prob=0.1,
        num_labels=1,
        pooling=pooling,
        pooling_kwargs={
            "num_heads": 4,
            "num_latents": 4,
            "dropout": 0.0,
            "attention_size": 16,
            "attention_hops": 4,
            "attention_dropout": 0.0,
            "fft_latent_gate_num_heads": 4,
            "fft_latent_gate_num_latents": 4,
        },
    )
    head = ClassificationHead2(cfg)
    features = torch.randn(2, 12, 64)
    mask = torch.ones(2, 12)
    out = head(features, mask)
    assert out.shape == (2, 1)

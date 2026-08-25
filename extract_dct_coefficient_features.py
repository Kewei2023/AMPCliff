#!/usr/bin/env python3
# maintained by kewei li
"""Exp5 / DC validation design v2 — Step 2: extract last-layer DCT coefficients.
Extract last-layer DCT coefficients from FLaG checkpoints for DC validation."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AMPCliff.factory.initializer import ModelInitializer
from AMPCliff.spectrual_filter.filter_seq import dct_ortho
from AMPCliff.utils.utils import fix_random_seed, load_model

DEFAULT_CONFIG_DIR = "/data/public/models/facebook/esm2_t6_8M_UR50D/"


def build_eval_cfg(
    *,
    model_version: str = "esm2_t6",
    pooling: str = "FLaG",
    config_dir: str = DEFAULT_CONFIG_DIR,
    max_length: int = 30,
) -> object:
    cfg = OmegaConf.create(
        {
            "task": {"type": "regression"},
            "features": {"type": "LLM"},
            "data": {"max_length": max_length},
            "model": {
                "config_dir": config_dir,
                "regression": {
                    "version": model_version,
                    "pooling": pooling,
                    "apply": "none",
                    "pooling_common": {
                        "num_anchor": 8,
                        "use_fft": True,
                        "num_heads": 4,
                        "gated": True,
                        "dropout": 0.0,
                        "num_latents": 8,
                        "analysis_dim": 8,
                    },
                    "pooling_config": {},
                    "check_point": {"load": False, "path": ""},
                },
            },
            "train": {"device_ids": [0], "random_seed": 0},
            "mode": {"ddp": False},
        }
    )
    return cfg


def _resolve_checkpoint_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    if checkpoint.is_file() and checkpoint.name == "model.pth":
        return checkpoint.parent.parent
    if (checkpoint / "data" / "model.pth").is_file():
        return checkpoint
    matches = list(checkpoint.rglob("model.pth"))
    if len(matches) == 1:
        return matches[0].parent.parent
    if len(matches) > 1:
        matches.sort(key=lambda p: len(str(p)))
        return matches[0].parent.parent
    raise FileNotFoundError(f"Could not resolve checkpoint directory from {checkpoint}")


def _backbone_state_hash(model: torch.nn.Module) -> str:
    net = model.module if hasattr(model, "module") else model
    encoder = net.pretrain_model
    state = encoder.state_dict()
    keys = sorted(k for k in state if k.startswith("encoder.") or k.startswith("embeddings."))
    if not keys:
        keys = sorted(state.keys())
    payload = b"".join(state[k].detach().cpu().numpy().tobytes() for k in keys)
    return hashlib.sha256(payload).hexdigest()


def _extract_residue_hidden(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return residue-only hidden states [L, d], excluding CLS/EOS/padding."""
    valid_positions = attention_mask.bool().nonzero(as_tuple=False).squeeze(-1).tolist()
    if len(valid_positions) <= 2:
        raise ValueError("Sequence too short after tokenization")
    residue_positions = valid_positions[1:-1]
    if not residue_positions:
        raise ValueError("No residue tokens found between special tokens")
    return last_hidden[residue_positions]


def _dct_coefficients(hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
    coeff = dct_ortho(hidden, dim=0)
    length = hidden.shape[0]
    coeff_0 = coeff[0]
    coeff_0_norm = coeff_0 / math.sqrt(length) if length > 0 else coeff_0
    return {
        "coeff_0_raw": coeff_0,
        "coeff_0_norm": coeff_0_norm,
        "coeff_1": coeff[1] if coeff.shape[0] > 1 else torch.zeros_like(coeff_0),
        "coeff_2": coeff[2] if coeff.shape[0] > 2 else torch.zeros_like(coeff_0),
        "coeff_3": coeff[3] if coeff.shape[0] > 3 else torch.zeros_like(coeff_0),
    }


@torch.no_grad()
def extract_features_for_sequences(
    model: torch.nn.Module,
    tokenizer,
    sequences: Sequence[str],
    idx_list: Sequence[int],
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 30,
) -> Dict[str, np.ndarray]:
    net = model.module if hasattr(model, "module") else model
    backbone = net.pretrain_model
    backbone.eval()

    coeff_0_raw: List[np.ndarray] = []
    coeff_0_norm: List[np.ndarray] = []
    coeff_1: List[np.ndarray] = []
    coeff_2: List[np.ndarray] = []
    coeff_3: List[np.ndarray] = []
    out_idx: List[int] = []
    out_seq: List[str] = []
    out_len: List[int] = []

    for start in range(0, len(sequences), batch_size):
        batch_seq = sequences[start : start + batch_size]
        batch_idx = idx_list[start : start + batch_size]
        spaced = [" ".join(seq) for seq in batch_seq]
        encoded = tokenizer(
            spaced,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = backbone(**encoded)
        hidden = outputs.last_hidden_state
        for i, seq in enumerate(batch_seq):
            h_res = _extract_residue_hidden(hidden[i], encoded["attention_mask"][i])
            coeffs = _dct_coefficients(h_res)
            out_idx.append(int(batch_idx[i]))
            out_seq.append(seq)
            out_len.append(int(h_res.shape[0]))
            coeff_0_raw.append(coeffs["coeff_0_raw"].detach().cpu().numpy().astype(np.float32))
            coeff_0_norm.append(coeffs["coeff_0_norm"].detach().cpu().numpy().astype(np.float32))
            coeff_1.append(coeffs["coeff_1"].detach().cpu().numpy().astype(np.float32))
            coeff_2.append(coeffs["coeff_2"].detach().cpu().numpy().astype(np.float32))
            coeff_3.append(coeffs["coeff_3"].detach().cpu().numpy().astype(np.float32))

    return {
        "idx": np.asarray(out_idx, dtype=np.int64),
        "sequence": np.asarray(out_seq, dtype=object),
        "length": np.asarray(out_len, dtype=np.int32),
        "coeff_0_raw": np.stack(coeff_0_raw, axis=0),
        "coeff_0_norm": np.stack(coeff_0_norm, axis=0),
        "coeff_1": np.stack(coeff_1, axis=0),
        "coeff_2": np.stack(coeff_2, axis=0),
        "coeff_3": np.stack(coeff_3, axis=0),
    }


def load_property_subset(
    property_table: Path,
    species: str,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_csv(property_table)
    sub = df[df["species"] == species].copy()
    if max_samples is not None and max_samples > 0:
        sub = sub.head(max_samples)
    return sub.reset_index(drop=True)


def save_npz_bundle(path: Path, bundle: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--property-table", type=Path, required=True)
    ap.add_argument("--species", choices=["e_coli", "s_aureus"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-version", default="esm2_t6")
    ap.add_argument("--pooling", default="FLaG")
    ap.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=30)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument(
        "--compare-backbone-seeds",
        action="store_true",
        help="Print backbone hash only; useful for deduplicating across seeds.",
    )
    args = ap.parse_args()

    fix_random_seed(args.random_seed, cuda_deterministic=True)
    ckpt_dir = _resolve_checkpoint_dir(args.checkpoint)
    cfg = build_eval_cfg(
        model_version=args.model_version,
        pooling=args.pooling,
        config_dir=args.config_dir,
        max_length=args.max_length,
    )
    device = torch.device(args.device)
    initializer = ModelInitializer(cfg, device)
    model, tokenizer = initializer.init()
    model = load_model(model, str(ckpt_dir), device)
    model.eval()

    if args.compare_backbone_seeds:
        print(_backbone_state_hash(model))
        return 0

    subset = load_property_subset(
        args.property_table,
        args.species,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    bundle = extract_features_for_sequences(
        model,
        tokenizer,
        subset["sequence"].tolist(),
        subset["idx"].tolist(),
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    save_npz_bundle(args.output, bundle)
    print(
        f"Wrote {args.output} | species={args.species} n={bundle['idx'].shape[0]} "
        f"hidden_dim={bundle['coeff_0_raw'].shape[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

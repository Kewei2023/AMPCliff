import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Optional

from AMPCliff.utils.std_logger import Logger
from AMPCliff.utils.distribution import setup_multinodes, cleanup_multinodes
from AMPCliff.loader.utils import make_loader
from AMPCliff.utils.utils import get_device, fix_random_seed, load_model
from AMPCliff.features.feature_fetcher import FeatureFetcher
from AMPCliff.factory.initializer import ModelInitializer
from AMPCliff.utils.path_helper import resolve_path
from AMPCliff.utils.fftlag_manifest_utils import (
    get_analysis_filter,
    idx_list_from_name2id,
    load_peptide_manifest,
)
from AMPCliff.utils.fftlag_latent_viz import (
    compute_attn_band_deviation,
    compute_gate_input_band_profiles,
    compute_gate_input_compare,
    compute_gate_input_stats,
    compute_latent_out_contribution,
    compute_query_band_mass,
    compute_query_diversity,
    compute_weighted_band_readout,
    gate_input_mode_from_pooling,
    render_primary_plots_per_sample,
    summarize_latent_query_discriminability,
    DEFAULT_PRIMARY_PLOTS,
)


def find_fft_gate_pooling(model: nn.Module) -> Optional[nn.Module]:
    from AMPCliff.factory.pooling.spectral_anchor_v2 import (
        FFTLatentAttentionGatePooling,
        FFTLatentAttentionGateV3Pooling,
    )

    for name, module in model.named_modules():
        if isinstance(module, (FFTLatentAttentionGatePooling, FFTLatentAttentionGateV3Pooling)):
            Logger.info(f"Found FFT latent gate pooling at: {name} ({type(module).__name__})")
            return module
    return None


@hydra.main(config_path="configs", config_name="evaluate_fftlag_mechanism.yaml")
def main(cfg: DictConfig):
    local_rank = 0
    global_rank = 0

    if cfg.mode.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        global_rank = int(os.environ["RANK"])
        setup_multinodes(local_rank, world_size)
        device = torch.device("cuda", local_rank)
        random_seed = cfg.train.random_seed + local_rank
    else:
        random_seed = cfg.train.random_seed
        device = get_device(cfg)

    if global_rank == 0:
        Logger.info(f"setting random seed: {random_seed}")
    fix_random_seed(random_seed, cuda_deterministic=True)

    latent_cfg = cfg.get("latent_analysis", {})
    k_bands = latent_cfg.get("k_bands", 8)
    base = latent_cfg.get("base", 4)
    band_mode = latent_cfg.get("band_mode", "uniform")
    splits = latent_cfg.get("splits", ["test"])
    plot_legacy = bool(latent_cfg.get("plot_legacy_attention_mass", False))
    primary_plots = list(latent_cfg.get("primary_plots", list(DEFAULT_PRIMARY_PLOTS)))

    manifest_path_str = str(getattr(cfg.get("analysis", {}), "peptide_manifest", "") or "").strip()
    if manifest_path_str and global_rank == 0:
        allowed_idx_log, _ = load_peptide_manifest(manifest_path_str)
        Logger.info(f"[Exp4] peptide manifest: {manifest_path_str} idx={sorted(allowed_idx_log)}")

    model, tokenizer = ModelInitializer(cfg, device).init()
    vocab_dict = tokenizer.get_vocab()
    feature_fetcher = FeatureFetcher(cfg, tokenizer)

    if cfg.mode.ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    actual_model = model.module if isinstance(model, DDP) else model
    gate_pooling = find_fft_gate_pooling(actual_model)
    if gate_pooling is None:
        Logger.error("FFTLatentAttentionGatePooling / V3Pooling not found.")
        return

    pooling_name = str(cfg.model.regression.pooling)
    gate_input_mode = gate_input_mode_from_pooling(pooling_name)
    Logger.info(f"gate_input_mode={gate_input_mode} (pooling={pooling_name})")

    if not cfg.model[cfg.task.type].check_point.load:
        Logger.error("please set model.regression.check_point.load=true and provide path")
        return
    model = load_model(model, cfg.model[cfg.task.type].check_point.path, device)
    actual_model = model.module if isinstance(model, DDP) else model
    gate_pooling = find_fft_gate_pooling(actual_model)
    if gate_pooling is None:
        Logger.error("FFTLatentAttentionGatePooling / V3Pooling not found after loading checkpoint.")
        return

    threshold = str(cfg.data.threshold)
    allowed_idx, max_samples = get_analysis_filter(cfg)
    if allowed_idx is not None and max_samples is None:
        max_samples = len(allowed_idx)

    for condition in cfg.data[cfg.task.type].condition:
        for diff in cfg.data.diff:
            if cfg.data[cfg.task.type].mode != "fix":
                continue

            test_file_path = resolve_path(
                cfg.data[cfg.task.type].fix.test_file,
                diff=diff,
                condition=condition,
                threshold=threshold,
                dataset=cfg.data[cfg.task.type].get("dataset"),
            )
            test_loader = make_loader(
                local_rank,
                test_file_path,
                cfg=cfg,
                batch_size=cfg.train.batch_size,
                vocab_dict=vocab_dict,
                pin_memory=False,
                num_workers=cfg.train.num_workers,
                random_seed=random_seed,
            )

            for split in splits:
                if split != "test":
                    Logger.warning(f"Exp4 currently supports test split; got {split}")
                loader = test_loader
                model.eval()

                records = []
                mass_frames: List[pd.DataFrame] = []
                dev_frames: List[pd.DataFrame] = []
                readout_frames: List[pd.DataFrame] = []
                contrib_frames: List[pd.DataFrame] = []
                gate_stats_rows: List[Dict] = []
                gate_compare_rows: List[Dict] = []
                gate_band_profile_frames: List[pd.DataFrame] = []
                diversity_rows: List[Dict] = []
                discriminability_rows: List[Dict] = []
                latent_out_by_idx: Dict[int, torch.Tensor] = {}
                attn_weights_by_idx: Dict[int, torch.Tensor] = {}
                freq_gate_module = getattr(gate_pooling, "freq_gate", None)
                seen = 0
                collected_idx = set()

                with torch.no_grad():
                    for data in loader:
                        sequence, name2id, label = data
                        batch_idxs = idx_list_from_name2id(name2id)
                        if allowed_idx is not None:
                            manifest_idxs = [i for i in batch_idxs if i in allowed_idx]
                            if not manifest_idxs:
                                continue
                            if all(i in collected_idx for i in manifest_idxs):
                                continue
                            collected_idx.update(manifest_idxs)
                        if max_samples is not None and seen >= max_samples:
                            break

                        token_sequence = feature_fetcher.query_features(sequence["peptide"])
                        batch1 = {k: torch.tensor(v).to(device) for k, v in token_sequence.items()}
                        actual_model(batch1)

                        if not hasattr(gate_pooling, "_last_latent_attn_weights"):
                            Logger.error("Pooling module missing _last_latent_attn_weights")
                            return

                        attn_w = gate_pooling._last_latent_attn_weights.cpu()
                        batch_size = attn_w.shape[0]
                        latent_out = getattr(gate_pooling, "_last_latent_out", None)
                        latent_summary = getattr(gate_pooling, "_last_latent_summary", None)
                        freq_tokens = getattr(gate_pooling, "_last_freq_tokens", None)
                        raw_gate = getattr(gate_pooling, "_last_raw_gate", None)

                        if latent_out is not None:
                            latent_out = latent_out.cpu()
                            if latent_out.dim() == 2:
                                latent_out = latent_out.unsqueeze(0)
                        if latent_summary is not None:
                            latent_summary = latent_summary.cpu()
                            if latent_summary.dim() == 1:
                                latent_summary = latent_summary.unsqueeze(0)
                        if freq_tokens is not None:
                            freq_tokens = freq_tokens.cpu()
                        if raw_gate is not None:
                            raw_gate = raw_gate.cpu()

                        for bi, idx_val in enumerate(batch_idxs):
                            if allowed_idx is not None and idx_val not in allowed_idx:
                                continue
                            if bi >= batch_size:
                                break
                            aw = attn_w[bi]
                            lo = latent_out[bi] if latent_out is not None else None
                            ls = latent_summary[bi] if latent_summary is not None else None
                            ft = freq_tokens[bi] if freq_tokens is not None else None
                            rg = raw_gate[bi] if raw_gate is not None else None

                            records.append({
                                "idx": idx_val,
                                "attn_weights": aw,
                                "latent_out": lo,
                                "latent_summary": ls,
                            })
                            attn_weights_by_idx[int(idx_val)] = aw

                            mass_df = compute_query_band_mass(aw, k_bands, base, band_mode)
                            mass_df["idx"] = idx_val
                            mass_frames.append(mass_df)

                            dev_df = compute_attn_band_deviation(aw, k_bands, base, band_mode)
                            dev_df["idx"] = idx_val
                            dev_frames.append(dev_df)

                            if ft is not None:
                                readout_df = compute_weighted_band_readout(
                                    aw, ft, k_bands, base, band_mode
                                )
                                readout_df["idx"] = idx_val
                                readout_frames.append(readout_df)
                            else:
                                Logger.warning(
                                    f"idx={idx_val}: missing _last_freq_tokens; skip weighted readout"
                                )

                            if lo is not None:
                                contrib_df = compute_latent_out_contribution(
                                    lo, mode=gate_input_mode
                                )
                                contrib_df["idx"] = idx_val
                                contrib_frames.append(contrib_df)
                                latent_out_by_idx[int(idx_val)] = lo

                                gate_row = compute_gate_input_stats(
                                    lo, rg, mode=gate_input_mode
                                )
                                gate_row["idx"] = idx_val
                                gate_stats_rows.append(gate_row)

                                discriminability_rows.append(
                                    summarize_latent_query_discriminability(
                                        lo,
                                        aw,
                                        idx_val,
                                        k_bands,
                                        base,
                                        band_mode,
                                        mode=gate_input_mode,
                                    )
                                )

                                if ft is not None and freq_gate_module is not None:
                                    cmp_row = compute_gate_input_compare(
                                        lo,
                                        ft,
                                        freq_gate_module,
                                        rg,
                                        mode=gate_input_mode,
                                    )
                                    cmp_row["idx"] = idx_val
                                    gate_compare_rows.append(cmp_row)

                                    band_prof = compute_gate_input_band_profiles(
                                        aw, ft, k_bands, base, band_mode
                                    )
                                    band_prof["idx"] = idx_val
                                    gate_band_profile_frames.append(band_prof)

                            diversity_rows.append(
                                compute_query_diversity(aw, idx_val, k_bands, base, band_mode)
                            )

                            seen += 1
                            if max_samples is not None and seen >= max_samples:
                                break
                        if allowed_idx is not None and len(collected_idx) >= len(allowed_idx):
                            break

                if global_rank != 0:
                    continue

                if not records:
                    Logger.warning("No latent records collected.")
                    continue

                torch.save({"samples": records}, "./latent_attn_weights.pt")
                Logger.info(f"[saved] ./latent_attn_weights.pt ({len(records)} samples)")

                mass_all = pd.concat(mass_frames, ignore_index=True)
                mass_all.to_csv("./latent_query_band_mass.csv", index=False)
                Logger.info("[saved] ./latent_query_band_mass.csv")

                dev_all = pd.concat(dev_frames, ignore_index=True)
                dev_all.to_csv("./latent_attn_band_deviation.csv", index=False)
                Logger.info("[saved] ./latent_attn_band_deviation.csv")

                if readout_frames:
                    readout_all = pd.concat(readout_frames, ignore_index=True)
                    readout_all.to_csv("./latent_weighted_band_readout.csv", index=False)
                    Logger.info("[saved] ./latent_weighted_band_readout.csv")
                else:
                    readout_all = None

                if contrib_frames:
                    contrib_all = pd.concat(contrib_frames, ignore_index=True)
                    contrib_all.to_csv("./latent_query_contribution.csv", index=False)
                    Logger.info("[saved] ./latent_query_contribution.csv")
                else:
                    contrib_all = None

                if gate_stats_rows:
                    gate_stats_df = pd.DataFrame(gate_stats_rows)
                    if len(gate_stats_df) > 1:
                        for col in gate_stats_df.columns:
                            if col == "idx":
                                continue
                            gate_stats_df[f"{col}_cross_sample_std"] = gate_stats_df[col].std()
                    gate_stats_df.to_csv("./latent_gate_input_stats.csv", index=False)
                    Logger.info("[saved] ./latent_gate_input_stats.csv")

                gate_compare_all = None
                if gate_compare_rows:
                    gate_compare_all = pd.DataFrame(gate_compare_rows)
                    if len(gate_compare_all) > 1:
                        for col in gate_compare_all.columns:
                            if col == "idx":
                                continue
                            gate_compare_all[f"{col}_cross_sample_std"] = gate_compare_all[col].std()
                    gate_compare_all.to_csv("./latent_gate_input_compare.csv", index=False)
                    Logger.info("[saved] ./latent_gate_input_compare.csv")

                gate_band_all = None
                if gate_band_profile_frames:
                    gate_band_all = pd.concat(gate_band_profile_frames, ignore_index=True)
                    gate_band_all.to_csv("./latent_gate_input_band_profile.csv", index=False)
                    Logger.info("[saved] ./latent_gate_input_band_profile.csv")

                div_df = pd.DataFrame(diversity_rows)
                if len(div_df) > 1:
                    for col in ("mean_query_cosine_distance", "mean_query_js_divergence", "query_attn_std_mean"):
                        div_df[f"{col}_cross_sample_std"] = div_df[col].std()
                div_df.to_csv("./latent_query_diversity.csv", index=False)
                Logger.info("[saved] ./latent_query_diversity.csv")

                if discriminability_rows:
                    disc_df = pd.DataFrame(discriminability_rows)
                    if len(disc_df) > 1:
                        mean_row = {"idx": -1}
                        for col in disc_df.columns:
                            if col == "idx":
                                continue
                            if pd.api.types.is_numeric_dtype(disc_df[col]):
                                mean_row[col] = float(disc_df[col].mean())
                        disc_df = pd.concat(
                            [disc_df, pd.DataFrame([mean_row])], ignore_index=True
                        )
                    disc_df.to_csv("./latent_query_discriminability.csv", index=False)
                    Logger.info("[saved] ./latent_query_discriminability.csv")

                dataset_label = str(cfg.data.regression.dataset)
                if contrib_all is not None and readout_all is not None and latent_out_by_idx:
                    render_primary_plots_per_sample(
                        contrib_df=contrib_all,
                        dev_df=dev_all,
                        readout_df=readout_all,
                        latent_out_by_idx=latent_out_by_idx,
                        dataset_label=dataset_label,
                        out_base=".",
                        primary_plots=primary_plots,
                        compare_df=gate_compare_all,
                        band_profile_df=gate_band_all,
                        mass_df=mass_all,
                        attn_weights_by_idx=attn_weights_by_idx,
                        plot_legacy=plot_legacy,
                        gate_input_mode=gate_input_mode,
                    )
                else:
                    Logger.warning(
                        "Skipped per-sample plots: need latent_out, freq_tokens, and readout data"
                    )

    if cfg.mode.ddp:
        cleanup_multinodes()


if __name__ == "__main__":
    main()
    print("done.")

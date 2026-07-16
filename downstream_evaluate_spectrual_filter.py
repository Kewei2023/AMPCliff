# maintained by kewei li
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import hydra
from pathlib import Path
from omegaconf import DictConfig
from AMPCliff.utils.std_logger import Logger
from AMPCliff.utils.scheduler import lr_scheduler
from AMPCliff.utils.distribution import setup_multinodes, cleanup_multinodes
from AMPCliff.loader.utils import make_loader
from AMPCliff.loader.split import random_split_data, stratified_split_data, fixed_cluster_split_data
from AMPCliff.utils.utils import get_device, fix_random_seed, load_weights, load_model
from AMPCliff.utils.evaluator import Evaluator
from AMPCliff.features.feature_fetcher import FeatureFetcher
from AMPCliff.factory.initializer import ModelInitializer
from AMPCliff.utils.path_helper import resolve_path
from AMPCliff.utils.metrics import Metrics
from AMPCliff.visualization.plot import plot_low_dimension
from AMPCliff.utils.match import seq2pair
from AMPCliff.spectrual_filter.filter import HiddenDimPrismHookHF as HiddenDimPrismHookHF_hidden
from AMPCliff.spectrual_filter.filter_seq import HiddenDimPrismHookHF as HiddenDimPrismHookHF_seq
from AMPCliff.spectrual_filter.rotary_knock import DisableRoPEHookHF   # ← 只对 hidden_dim 做频带处理
from AMPCliff.spectrual_filter.hidden_energy import band_energy_share_per_layer_4d,plot_band_shares_all_layers,plot_mse_diff_in_groups
from AMPCliff.utils.fftlag_manifest_utils import load_peptide_manifest
import mlflow
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
import math

# ------------------------- 小工具：收集/计算 MSE -------------------------
def _gather_if_ddp(y_pred, y_true, ids,all_layer_features, cfg, world_size, global_rank):
    """把 evaluator 里得到的 y_pred/y_true/ids 在 DDP 下做 all_gather；非 DDP 原样返回。"""
    if not cfg.mode.ddp:
        return y_pred, y_true, ids,all_layer_features
    gather_list_pred                = [torch.zeros_like(y_pred) for _ in range(world_size)]
    gather_list_true                = [torch.zeros_like(y_true) for _ in range(world_size)]
    gather_list_ids                 = [torch.zeros_like(ids)    for _ in range(world_size)]
    gather_list_all_layer_features  = [torch.zeros_like(all_layer_features)    for _ in range(world_size)]
    
    torch.distributed.all_gather(gather_list_pred, y_pred)
    torch.distributed.all_gather(gather_list_true, y_true)
    torch.distributed.all_gather(gather_list_ids,  ids)
    torch.distributed.all_gather(gather_list_all_layer_features,  all_layer_features)
    if global_rank == 0:
        y_pred             = torch.cat(gather_list_pred)
        y_true             = torch.cat(gather_list_true)
        ids                = torch.cat(gather_list_ids)
        all_layer_features = torch.cat(gather_list_all_layer_features)
    return y_pred, y_true, ids, all_layer_features

@torch.no_grad()
def evaluate_split_and_mse(evaluator, split, cfg, world_size, global_rank):
    """运行 evaluator.run(split)，返回 (metrics_dict, mse_scalar, y_pred, y_true, ids, all_layer_features, per_sample_mse)。"""
    metrics = evaluator.run(split)
    y_pred, y_true, ids, all_layer_features = evaluator.y_pred, evaluator.y_true, evaluator.ids, evaluator.all_layer_features
    y_pred, y_true, ids,all_layer_features = _gather_if_ddp(y_pred, y_true, ids,all_layer_features, cfg, world_size, global_rank)
    # 只在 rank0 计算与返回；其他 rank 返回占位
    if (not cfg.mode.ddp) or (global_rank == 0):
        per_sample_mse = (y_pred.squeeze(-1) - y_true) ** 2
        mse = per_sample_mse.mean().item()
        return metrics, mse, y_pred, y_true, ids, all_layer_features, per_sample_mse
    else:
        return metrics, None, None, None, None, None, None


def _get_encoder(model):
    net = model.module if hasattr(model, "module") else model
    return net.pretrain_model.encoder


def _make_prism_hook(dim, encoder, target_layers, band_index, k_bands, base, band_mode, mode, preserve_norm):
    if dim == "seq_len":
        Logger.info(f"Using seq_len DCT hook (band_index={band_index})")
        return HiddenDimPrismHookHF_seq(
            hf_esm=encoder,
            target_layers=target_layers,
            band_index=band_index,
            k_bands=k_bands,
            base=base,
            mode=mode,
            preserve_norm=preserve_norm,
        )
    Logger.info(f"Using hidden_dim DCT hook (band_index={band_index})")
    return HiddenDimPrismHookHF_hidden(
        hf_esm=encoder,
        target_layers=target_layers,
        band_index=band_index,
        k_bands=k_bands,
        base=base,
        band_mode=band_mode,
        mode=mode,
        preserve_norm=preserve_norm,
    )

# ======================================================================

@hydra.main(config_path="configs", config_name="evaluate_spectrual.yaml")
def main(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    cfg.orig_cwd = orig_cwd
    local_rank = 0
    global_rank = 0

    if not cfg.mode.nni and cfg.logger.log:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    if cfg.mode.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        global_rank = int(os.environ['RANK'])
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_SHM_DISABLE'] = '1'
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        random_seed = cfg.train.random_seed + local_rank
        setup_multinodes(local_rank, world_size)
        device = torch.device("cuda", local_rank)
    else:
        world_size = 1
        random_seed = cfg.train.random_seed
        device = get_device(cfg)

    if global_rank == 0:
        Logger.info("setting random seed: {}".format(random_seed))
    fix_random_seed(random_seed, cuda_deterministic=True)

    if cfg.logger.log and global_rank == 0:
        for p, v in cfg.mode.items(): mlflow.log_param(p, v)
        for p, v in cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].items():
            try: mlflow.log_param(p, v)
            except: mlflow.log_param(p, v[:500])
        for p, v in cfg.train.items(): mlflow.log_param(p, v)
        for p, v in cfg.model[cfg.task.type].items(): mlflow.log_param(p, v)

    metric_func = Metrics(cfg.task.type, topK=50)
    model, tokenizer = ModelInitializer(cfg, device).init()
    vocab_dict = tokenizer.get_vocab()
    feature_fetcher = FeatureFetcher(cfg, tokenizer)

    if cfg.mode.ddp:
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True)

    threshold = str(cfg.data.threshold)
    SPLIT_FOR_PLOTS = cfg.model.regression.spectrual_filter.split_for_plot # ← 若要看 valid，把它改成 'valid'

    # 频带与模式（可通过 cfg 传入；这里给默认：k_bands=5, base=4, notch 高频 band_index=4）
    k_bands    = cfg.model.regression.spectrual_filter.k_bands
    base       = cfg.model.regression.spectrual_filter.base
    band_mode  = cfg.model.regression.spectrual_filter.band_mode
    band_index_list = cfg.model.regression.spectrual_filter.band_index      # 0=LOW ... 4=HIGH
    mode       = cfg.model.regression.spectrual_filter.mode  # or 'pass'
    preserve_norm = cfg.model.regression.spectrual_filter.preserve_norm # true
    dim        = cfg.model.regression.spectrual_filter.dim
    manifest_path_str = str(getattr(cfg.get("analysis", {}), "peptide_manifest", "") or "").strip()
    if manifest_path_str and global_rank == 0:
        allowed_idx, manifest_meta = load_peptide_manifest(manifest_path_str)
        Logger.info(f"[Exp1] peptide manifest: {manifest_path_str} idx={sorted(allowed_idx)}")

    for condition in cfg.data[cfg.task.type].condition:
        for diff in cfg.data.diff:
            # --------------------- 加载数据 ---------------------
            if cfg.data[cfg.task.type].mode == 'fix':
                if global_rank == 0: Logger.info('loading datasets...')
                # def _mk(path_tmpl):
                #     return path_tmpl.replace("{diff}", str(diff)).replace("{condition}", condition).replace("{threshold}", threshold)
                # train_file_path = _mk(cfg.data[cfg.task.type].fix.train_file)
                train_file_path = resolve_path(
                                        cfg.data[cfg.task.type].fix.train_file,
                                        diff=diff,
                                        condition=condition,
                                        threshold=threshold,
                                        dataset=cfg.data[cfg.task.type].get('dataset')
                                    )
                # valid_file_path = _mk(cfg.data[cfg.task.type].fix.valid_file)
                valid_file_path = resolve_path(
                                        cfg.data[cfg.task.type].fix.valid_file,
                                        diff=diff,
                                        condition=condition,
                                        threshold=threshold,
                                        dataset=cfg.data[cfg.task.type].get('dataset')
                                    )
                # test_file_path  = _mk(cfg.data[cfg.task.type].fix.test_file)
                test_file_path = resolve_path(
                                        cfg.data[cfg.task.type].fix.test_file,
                                        diff=diff,
                                        condition=condition,
                                        threshold=threshold,
                                        dataset=cfg.data[cfg.task.type].get('dataset')
                                    )
                train_loader = make_loader(local_rank, train_file_path, cfg=cfg, batch_size=cfg.train.batch_size, vocab_dict=vocab_dict,
                                           pin_memory=False, num_workers=cfg.train.num_workers, random_seed=random_seed)
                valid_loader = make_loader(local_rank, valid_file_path, cfg=cfg, batch_size=cfg.train.batch_size, vocab_dict=vocab_dict,
                                           pin_memory=False, num_workers=cfg.train.num_workers, random_seed=random_seed)
                test_loader  = make_loader(local_rank,  test_file_path, cfg=cfg, batch_size=cfg.train.batch_size, vocab_dict=vocab_dict,
                                           pin_memory=False, num_workers=cfg.train.num_workers, random_seed=random_seed)
                dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

                # --------------------- 加载 checkpoint ---------------------
                if global_rank == 0: Logger.info('loading checkpoint...')
                if not cfg.model[cfg.task.type].check_point.load:
                    Logger.info('please give a check point path and set "load" to true'); return
                else:
                    model = load_model(model, cfg.model[cfg.task.type].check_point.path, device)

                evaluator = Evaluator(model, dataloaders, metric_func, feature_fetcher, device, cfg)

                
                # ===================== 0) 选择数据集=====================
                for SPLIT_FOR_PLOT in SPLIT_FOR_PLOTS:
                    # if L in cfg.model.regression.spectrual_filter.rotary_knock:
                    
                    # ===================== 1) 基线（无 hook）MSE =====================
                    if global_rank == 0: Logger.info(f"computing baseline MSE on {SPLIT_FOR_PLOT} set ...")
                    _, mse_base, y_pred, y_true, ids, all_layer_features, per_sample_mse_base = evaluate_split_and_mse(
                        evaluator, SPLIT_FOR_PLOT, cfg, world_size, global_rank
                    )

                    num_layers = _get_encoder(model).layer.__len__()
                    layer_ids = list(range(num_layers))
                    mse_diffs = {int(b): [] for b in band_index_list}   # difference = MSE(with hook) - MSE_base
                    per_sample_rows = []

                    
                    if global_rank == 0:
                        Logger.info(f"Baseline MSE ({SPLIT_FOR_PLOT}): {mse_base:.6f}")

                    df_long, edges = band_energy_share_per_layer_4d(
                                                                    X=all_layer_features[...,-1].unsqueeze(-1),
                                                                    token_valid_mask=None,   # 或 None
                                                                    batch_size=64,           # 视显存/内存调整
                                                                    device=device,            # 或 "cpu"
                                                                    dim=dim
                                                                )

                    plot_band_shares_all_layers(
                        df_long, edges,
                        k_bands=k_bands,
                        base=base,
                        band_mode=band_mode,
                        title=f"Hidden-DCT band energy share (k={k_bands}, base={base})",
                        out_png=f"./band_energy_share_layers_{SPLIT_FOR_PLOT}.png",
                        out_csv_curve=f"./band_energy_share_layers_curve_{SPLIT_FOR_PLOT}.csv"
                    )

                    
                    # ===================== 2) 循环band index =====================
                    for band_index in band_index_list:
                        # mse_diffs[f"band index {band_index}"] = []
                        if global_rank == 0:
                            Logger.info(f"start layer-wise spectral {mode}: band_index={band_index}, k={k_bands}, base={base}")

                        # ===================== 3) 逐层频带敲除 =====================
                        for L in layer_ids:
                            hook = _make_prism_hook(
                                dim=dim,
                                encoder=_get_encoder(model),
                                target_layers=[L],
                                band_index=band_index,
                                k_bands=k_bands,
                                base=base,
                                band_mode=band_mode,
                                mode=mode,
                                preserve_norm=preserve_norm,
                            )
                            hook.register()

                            _, mse_L, _, _, ids_L, _, per_sample_mse_L = evaluate_split_and_mse(
                                evaluator, SPLIT_FOR_PLOT, cfg, world_size, global_rank
                            )
                            hook.remove()

                            if (not cfg.mode.ddp) or (global_rank == 0):
                                diff_val = mse_L - mse_base
                                mse_diffs[int(band_index)].append(diff_val)
                                Logger.info(f"Layer {L:02d}: MSE={mse_L:.6f}, diff={diff_val:+.6f}")
                                for i in range(len(ids_L)):
                                    idx_val = int(ids_L[i].item())
                                    mse_b = float(per_sample_mse_base[i].item())
                                    mse_h = float(per_sample_mse_L[i].item())
                                    per_sample_rows.append({
                                        "idx": idx_val,
                                        "layer": L,
                                        "band": int(band_index),
                                        "split": SPLIT_FOR_PLOT,
                                        "mse_base": mse_b,
                                        "mse_with_hook": mse_h,
                                        "mse_diff": mse_h - mse_b,
                                    })

                    # ===================== 3) 可视化：层 vs MSE difference =====================
                    if (not cfg.mode.ddp) or (global_rank == 0):
                        rows = []
                        for b in band_index_list:
                            diffs_b = mse_diffs[int(b)]
                            # 防御性检查：确保每个 band 的列表长度与层数一致
                            # assert len(diffs_b) == len(layer_ids), f"band {b}: got {len(diffs_b)} diffs, expected {len(layer_ids)}"
                            for lid, val in zip(layer_ids, diffs_b):
                                rows.append({"layer": lid, "band": int(b), "mse_diff": float(val), "split": SPLIT_FOR_PLOT})
                        plot_df = pd.DataFrame(rows)  # columns: layer, band, mse_diff, split
                        # ipdb.set_trace()
                        # 画图（多条线）
                        # palette = sns.color_palette("tab20", n_colors=k_bands) 
                        # plt.figure(figsize=(10, 5))
                        # ax = sns.lineplot(data=plot_df, x="layer", y="mse_diff", hue="band", marker="o",palette=palette)
                        # ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
                        # ax.set_title(f"ESM2 spectral {mode} on {SPLIT_FOR_PLOT} (k={k_bands}, base={base})\nMSE difference: (with-filter − baseline)")
                        # ax.set_xlabel("Layer index")
                        # ax.set_ylabel("MSE difference")
                        # ax.legend(title="band index")
                        # plt.tight_layout()
                        plot_mse_diff_in_groups(
                            plot_df,
                            k_bands=k_bands, group_size=5,
                            split=SPLIT_FOR_PLOT, k=k_bands, base=base,
                            out_prefix="./mse_diff_bandgroup_",
                            palette_name="tab10",
                            set_common_ylim=True
                        )
                        # 保存图与 CSV（宽表 + 长表各一份）
                        out_png = f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-{SPLIT_FOR_PLOT}-mse_diff-layer_multi-band.png'
                        plt.savefig(out_png, dpi=150)
                        Logger.info(f"saved figure: {out_png}")

                        # 长表直接保存
                        out_csv_long = out_png.replace(".png", "_long.csv")
                        plot_df.to_csv(out_csv_long, index=False)

                        # 宽表（每个 band 一列），便于后续查阅
                        wide = plot_df.pivot(index="layer", columns="band", values="mse_diff").sort_index(axis=1)
                        wide.columns = [f"band{int(c)}" for c in wide.columns]
                        wide = wide.reset_index()
                        out_csv_wide = out_png.replace(".png", "_wide.csv")
                        wide.to_csv(out_csv_wide, index=False)
                        Logger.info(f"saved csv (long & wide): {out_csv_long}, {out_csv_wide}")

                        if per_sample_rows:
                            ps_df = pd.DataFrame(per_sample_rows)
                            ps_df.to_csv("./per_sample_band_sensitivity.csv", index=False)
                            Logger.info(f"saved per-sample sensitivity: ./per_sample_band_sensitivity.csv ({len(ps_df)} rows)")

                # ===================== 4) 原有三份 split 的常规评估与落盘（保持你的逻辑） =====================
                _manifest_active = bool(manifest_path_str)
                if _manifest_active:
                    if global_rank == 0:
                        Logger.info(
                            "[Exp1] peptide manifest active: skip train/valid/full-test CSV export"
                        )
                elif global_rank == 0:
                    Logger.info("evaluating train set......")
                if not _manifest_active:
                    evaluate_tr_metrics, _, _, _, _, _, _ = evaluate_split_and_mse(evaluator, 'train', cfg, world_size, global_rank)
                if not _manifest_active and global_rank == 0:
                    metric_func(evaluator.y_pred, evaluator.y_true, split='train', plot=True)
                    df = pd.read_csv(train_file_path)
                    pred_df = pd.DataFrame.from_dict({
                        f'{cfg.model[cfg.task.type].version}-pred': evaluator.y_pred.squeeze(1).cpu(),
                        'true': evaluator.y_true.cpu(),
                        'Idx': evaluator.ids
                    })
                    res_df = pd.merge(df, pred_df, on='Idx', how='outer').sort_values(by='Idx')
                    res_df.to_csv(f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-train_result.csv')

                if not _manifest_active and global_rank == 0:
                    Logger.info("evaluating valid set......")
                if not _manifest_active:
                    evaluate_val_metrics, _, _, _, _, _, _ = evaluate_split_and_mse(evaluator, 'valid', cfg, world_size, global_rank)
                if not _manifest_active and global_rank == 0:
                    metric_func(evaluator.y_pred, evaluator.y_true, split='valid', plot=True)
                    df = pd.read_csv(valid_file_path)
                    pred_df = pd.DataFrame.from_dict({
                        f'{cfg.model[cfg.task.type].version}-pred': evaluator.y_pred.squeeze(1).cpu(),
                        'true': evaluator.y_true.cpu(),
                        'Idx': evaluator.ids
                    })
                    res_df = pd.merge(df, pred_df, on='Idx', how='outer').sort_values(by='Idx')
                    res_df.to_csv(f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-valid_result.csv')

                if not _manifest_active and global_rank == 0:
                    Logger.info("evaluating test set......")
                    Logger.info("and save pair info......")
                if not _manifest_active:
                    evaluate_test_metrics, _, _, _, _, _, _ = evaluate_split_and_mse(evaluator, 'test', cfg, world_size, global_rank)
                if not _manifest_active and global_rank == 0:
                    metric_func(evaluator.y_pred, evaluator.y_true, split='test', plot=True)
                    df = pd.read_csv(test_file_path)
                    pred_df = pd.DataFrame.from_dict({
                        f'{cfg.model[cfg.task.type].version}-pred': evaluator.y_pred.squeeze(1).cpu(),
                        'true': evaluator.y_true.cpu(),
                        'Idx': evaluator.ids
                    })
                    res_df = pd.merge(df, pred_df, on='Idx', how='outer').sort_values(by='Idx')
                    res_df.to_csv(f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-test_result.csv')

                if global_rank == 0 and (not cfg.mode.nni) and cfg.logger.log and not _manifest_active:
                    for metric_name, metric_v in evaluate_tr_metrics.items():
                        if isinstance(metric_v, (float, np.float64, int)):
                            mlflow.log_metric(f"train_final/{metric_name}", metric_v, step=1)
                        elif isinstance(metric_v, str):
                            mlflow.log_text(metric_v, "train_final/report.txt")
                    for metric_name, metric_v in evaluate_val_metrics.items():
                        if isinstance(metric_v, (float, np.float64, int)):
                            mlflow.log_metric(f"valid_final/{metric_name}", metric_v, step=1)
                        elif isinstance(metric_v, str):
                            mlflow.log_text(metric_v, "valid_final/report.txt")
                    for metric_name, metric_v in evaluate_test_metrics.items():
                        if isinstance(metric_v, (float, np.float64, int)):
                            mlflow.log_metric(f"test_final/{metric_name}", metric_v, step=1)
                        elif isinstance(metric_v, str):
                            mlflow.log_text(metric_v, "test_final/report.txt")

    if cfg.mode.ddp:
        cleanup_multinodes()

if __name__ == '__main__':
    main()
    print('done.')

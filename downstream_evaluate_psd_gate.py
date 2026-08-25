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
from AMPCliff.utils.distribution import setup_multinodes, cleanup_multinodes
from AMPCliff.loader.utils import make_loader
from AMPCliff.loader.split import random_split_data, stratified_split_data, fixed_cluster_split_data
from AMPCliff.utils.utils import get_device, fix_random_seed, load_weights, load_model
from AMPCliff.utils.evaluator import Evaluator
from AMPCliff.features.feature_fetcher import FeatureFetcher
from AMPCliff.factory.initializer import ModelInitializer
from AMPCliff.utils.path_helper import resolve_path
from AMPCliff.utils.metrics import Metrics
from AMPCliff.spectrual_filter.hidden_energy import band_energy_share_per_layer_4d
from AMPCliff.taichinet.visual import (
    fft_layer_freq_distribution,
    plot_psd_band_distribution,
    plot_psd_cumulative_curve,
    compute_band_energy_concentration,
)
import mlflow
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, List, Tuple
from AMPCliff.spectrual_filter.filter import allocate_prism_bands
from AMPCliff.utils.fftlag_manifest_utils import (
    get_analysis_filter,
    idx_list_from_name2id,
    load_peptide_manifest,
)


# ======================== Gate 权重收集工具 ========================

class GateWeightCollector:
    """
    通过 PyTorch forward hook 截取 FFTLatentAttentionGatePooling 的 gate 权重。

    Hook 注册到 freq_gate (nn.Sequential) 上，在其 forward 后截取 sigmoid 输出。
    同时从 pooling 模块的属性中读取 _last_freq_tokens 等中间变量。
    """

    def __init__(self):
        self._hooks = []
        self.raw_gates = []       # list of (B, 2D) tensors - sigmoid output
        self.freq_tokens = []     # list of (B, F, 2D) tensors
        self.enhanced_freq = []   # list of (B, F, 2D) tensors

    def register(self, pooling_module: nn.Module):
        """在 FFTLatentAttentionGatePooling.freq_gate 上注册 hook。"""
        hook = pooling_module.freq_gate.register_forward_hook(self._hook_fn)
        self._hooks.append(hook)
        self._pooling_module = pooling_module
        Logger.info(f"GateWeightCollector: registered hook on {type(pooling_module).__name__}.freq_gate")

    def _hook_fn(self, module, input, output):
        """freq_gate 的 forward hook -- output 即 sigmoid 的输入前的线性输出。
        但实际 gate = sigmoid(freq_gate(x))，sigmoid 在 _apply_gate 里调用。
        所以这个 hook 截取的是 freq_gate 内部最后一层 Linear 的输出。
        我们直接从 pooling_module 的 _last_raw_gate 读取。"""
        pass

    def collect_from_module(self):
        """从 pooling 模块收集最近一次 forward 产生的 gate 和 freq_tokens。"""
        mod = self._pooling_module
        if hasattr(mod, '_last_raw_gate'):
            self.raw_gates.append(mod._last_raw_gate.cpu())
        if hasattr(mod, '_last_freq_tokens'):
            self.freq_tokens.append(mod._last_freq_tokens.cpu())
        if hasattr(mod, '_last_enhanced_freq'):
            self.enhanced_freq.append(mod._last_enhanced_freq.cpu())

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """将收集到的所有 batch 的 gate 数据聚合。"""
        result = {}
        if self.raw_gates:
            result['raw_gate'] = torch.cat(self.raw_gates, dim=0)      # (N, 2D)
        if self.freq_tokens:
            result['freq_tokens'] = torch.cat(self.freq_tokens, dim=0)  # (N, F, 2D)
        if self.enhanced_freq:
            result['enhanced_freq'] = torch.cat(self.enhanced_freq, dim=0)  # (N, F, 2D)
        return result


def find_fft_gate_pooling(model: nn.Module) -> Optional[nn.Module]:
    """递归搜索模型中的 FFTLatentAttentionGatePooling 模块。"""
    from AMPCliff.factory.pooling.flag_pooling import FFTLatentAttentionGatePooling
    for name, module in model.named_modules():
        if isinstance(module, FFTLatentAttentionGatePooling):
            Logger.info(f"Found FFTLatentAttentionGatePooling at: {name}")
            return module
    return None


# ======================== Gate 权重分析函数 ========================

@torch.no_grad()
def compute_gate_weight_by_freq_band(
    gate_data: Dict[str, torch.Tensor],
    seq_len: int,
    k_bands: int = 8,
    base: int = 4,
    band_mode: str = "uniform",
) -> pd.DataFrame:
    """
    计算 gate 对各频率 bin 的有效影响，并按频段聚合。

    effective_weight[b, f] = |enhanced_freq[b, f]| / |freq_tokens[b, f]|
    即 gate 调制后与调制前的能量比。

    返回 DataFrame: band, band_start, band_end, mean_effective_weight, energy_before, energy_after
    """
    freq_tokens = gate_data['freq_tokens']     # (N, F, 2D)
    enhanced_freq = gate_data['enhanced_freq']  # (N, F, 2D)
    raw_gate = gate_data['raw_gate']            # (N, 2D)

    N, F, twoD = freq_tokens.shape
    assert twoD == raw_gate.shape[1]

    # 计算各频率 bin 的能量
    energy_before = (freq_tokens ** 2).sum(dim=-1)    # (N, F)
    energy_after = (enhanced_freq ** 2).sum(dim=-1)   # (N, F)

    # 有效权重 = energy_after / energy_before (clamped)
    eps = 1e-8
    effective_weight = energy_after / (energy_before + eps)  # (N, F)

    # 频段划分
    masks, sizes, starts, ends = allocate_prism_bands(F, k=k_bands, base=base, mode=band_mode)

    rows = []
    for bi in range(k_bands):
        s = int(starts[bi].item())
        e = int(ends[bi].item())
        ew_band = effective_weight[:, s:e].mean().item()
        eb_band = energy_before[:, s:e].mean().item()
        ea_band = energy_after[:, s:e].mean().item()
        rows.append({
            "band": bi,
            "band_start": s,
            "band_end": e,
            "mean_effective_weight": ew_band,
            "energy_before": eb_band,
            "energy_after": ea_band,
        })

    return pd.DataFrame(rows)


def compute_per_sample_gate_by_band(
    freq_tokens: torch.Tensor,
    enhanced_freq: torch.Tensor,
    idx: int,
    k_bands: int = 8,
    base: int = 4,
    band_mode: str = "uniform",
) -> pd.DataFrame:
    """Per-sample band-level gate energy stats."""
    ft = freq_tokens.unsqueeze(0) if freq_tokens.dim() == 2 else freq_tokens
    ef = enhanced_freq.unsqueeze(0) if enhanced_freq.dim() == 2 else enhanced_freq
    energy_before = (ft ** 2).sum(dim=-1)
    energy_after = (ef ** 2).sum(dim=-1)
    eps = 1e-8
    effective_weight = energy_after / (energy_before + eps)
    _, _, starts, ends = allocate_prism_bands(ft.shape[1], k=k_bands, base=base, mode=band_mode)
    rows = []
    for bi in range(k_bands):
        s = int(starts[bi].item())
        e = int(ends[bi].item())
        rows.append({
            "idx": idx,
            "band": bi,
            "band_start": s,
            "band_end": e,
            "energy_before": energy_before[:, s:e].mean().item(),
            "energy_after": energy_after[:, s:e].mean().item(),
            "effective_weight": effective_weight[:, s:e].mean().item(),
        })
    return pd.DataFrame(rows)


def _idx_list_from_name2id(name2id) -> List[int]:
    idxs: List[int] = []
    for _name, ids in name2id.items():
        if isinstance(ids, (list, tuple)):
            idxs.extend(int(x) for x in ids)
        else:
            idxs.append(int(ids))
    return idxs


def plot_gate_weight_distribution(
    gate_data: Dict[str, torch.Tensor],
    seq_len: int,
    k_bands: int = 8,
    base: int = 4,
    band_mode: str = "uniform",
    threshold: float = 0.9,
    title_prefix: str = "FFTLatentAttentionGate",
    out_dir: str = ".",
):
    """
    生成 Gate 权重分布的可视化图表。

    图1: Gate 权重 vs 频率 bin (effective weight per freq bin)
    图2: 累积 Gate 权重百分比曲线，标注 90% 阈值
    图3: Gate 调制前后各频段能量对比
    """
    freq_tokens = gate_data['freq_tokens']     # (N, F, 2D)
    enhanced_freq = gate_data['enhanced_freq']  # (N, F, 2D)
    raw_gate = gate_data['raw_gate']            # (N, 2D)

    N, F, twoD = freq_tokens.shape

    energy_before = (freq_tokens ** 2).sum(dim=-1).mean(dim=0).numpy()    # (F,)
    energy_after = (enhanced_freq ** 2).sum(dim=-1).mean(dim=0).numpy()   # (F,)

    eps = 1e-8
    effective_weight = (enhanced_freq ** 2).sum(dim=-1) / ((freq_tokens ** 2).sum(dim=-1) + eps)
    mean_ew = effective_weight.mean(dim=0).numpy()  # (F,)

    masks, sizes, starts, ends = allocate_prism_bands(F, k=k_bands, base=base, mode=band_mode)

    # ---- 图1: Effective weight vs frequency bin ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    ax.bar(range(F), mean_ew, color="steelblue", alpha=0.8)
    for bi in range(k_bands):
        s, e = int(starts[bi].item()), int(ends[bi].item())
        ax.axvline(s, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Frequency bin index (rfft along sequence)")
    ax.set_ylabel("Effective weight (energy ratio)")
    ax.set_title(f"{title_prefix}: Gate effective weight per freq bin")

    # ---- 图2: 累积 Gate 权重 ----
    ax2 = axes[1]
    total_energy_after = energy_after.sum()
    prob_after = energy_after / (total_energy_after + eps)
    cum_energy = np.cumsum(prob_after)
    ax2.plot(range(F), cum_energy, linewidth=2, color="darkorange")
    ax2.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"{threshold:.0%} threshold")
    for bi in range(k_bands):
        s, e = int(starts[bi].item()), int(ends[bi].item())
        ax2.axvline(s, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # 找到达到 90% 的频率 bin
    idx_90 = np.searchsorted(cum_energy, threshold)
    if idx_90 < F:
        ax2.axvline(idx_90, color="green", linestyle=":", linewidth=1.5,
                     label=f"90% reached at bin {idx_90}")
    ax2.set_xlabel("Frequency bin index")
    ax2.set_ylabel("Cumulative energy share (after gate)")
    ax2.set_title(f"{title_prefix}: Cumulative energy distribution")
    ax2.legend()

    # ---- 图3: Gate 前后能量对比 (按频段) ----
    ax3 = axes[2]
    band_eb = []
    band_ea = []
    band_labels = []
    for bi in range(k_bands):
        s, e = int(starts[bi].item()), int(ends[bi].item())
        band_eb.append(energy_before[s:e].sum())
        band_ea.append(energy_after[s:e].sum())
        band_labels.append(f"B{bi}\n[{s},{e})")

    x = np.arange(k_bands)
    width = 0.35
    ax3.bar(x - width / 2, band_eb, width, label="Before gate", color="steelblue", alpha=0.8)
    ax3.bar(x + width / 2, band_ea, width, label="After gate", color="darkorange", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(band_labels, fontsize=7)
    ax3.set_xlabel("Frequency band")
    ax3.set_ylabel("Mean energy")
    ax3.set_title(f"{title_prefix}: Energy before vs after gate")
    ax3.legend()

    plt.tight_layout()
    out_png = os.path.join(out_dir, "gate_weight_distribution.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png}")

    # ---- 额外: 累积曲线单独大图 ----
    plt.figure(figsize=(8, 5))
    plt.plot(range(F), cum_energy, linewidth=2, color="darkorange", marker="o", markersize=3)
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"{threshold:.0%} threshold")
    for bi in range(k_bands):
        s, e = int(starts[bi].item()), int(ends[bi].item())
        plt.axvline(s, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # 高亮关键频段
    key_bands_end = min(3, k_bands)
    x_lo = int(starts[0].item())
    x_hi = int(ends[key_bands_end - 1].item())
    plt.axvspan(x_lo, x_hi, alpha=0.1, color="red", label=f"Key bands [0,{key_bands_end})")

    plt.xlabel("Frequency bin index (rfft along sequence)")
    plt.ylabel("Cumulative energy share (after gate)")
    plt.title(f"{title_prefix}: Cumulative Gate energy distribution")
    plt.legend()
    plt.tight_layout()
    out_png_cum = os.path.join(out_dir, "gate_cumulative_curve.png")
    plt.savefig(out_png_cum, dpi=200, bbox_inches="tight")
    plt.close()
    Logger.info(f"[saved] {out_png_cum}")

    # ---- 保存数据 ----
    df_band = compute_gate_weight_by_freq_band(gate_data, seq_len, k_bands=k_bands, base=base, band_mode=band_mode)
    out_csv = os.path.join(out_dir, "gate_weight_by_band.csv")
    df_band.to_csv(out_csv, index=False)
    Logger.info(f"[saved] {out_csv}")

    # 打印关键统计
    total_after = df_band['energy_after'].sum()
    df_band['share_after'] = df_band['energy_after'] / total_after
    cum_share = df_band['share_after'].cumsum()
    Logger.info("=== Gate Energy Distribution by Band ===")
    for _, row in df_band.iterrows():
        bi = int(row['band'])
        Logger.info(f"  Band {bi} [{int(row['band_start'])},{int(row['band_end'])}): "
                     f"share={row['share_after']:.4f}, cum={cum_share[bi]:.4f}, "
                     f"eff_weight={row['mean_effective_weight']:.4f}")
    return df_band


# ======================== 主评估流程 ========================

@hydra.main(config_path="configs", config_name="evaluate_psd_gate.yaml")
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
        os.environ["NCCL_SOCKET_IFACE"] = "eth0"
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

    metric_func = Metrics(cfg.task.type, topK=50)
    model, tokenizer = ModelInitializer(cfg, device).init()
    vocab_dict = tokenizer.get_vocab()
    feature_fetcher = FeatureFetcher(cfg, tokenizer)

    if cfg.mode.ddp:
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True)

    # ---- 找到 FFTLatentAttentionGatePooling 模块 ----
    actual_model = model.module if isinstance(model, DDP) else model
    gate_pooling = find_fft_gate_pooling(actual_model)
    if gate_pooling is None:
        Logger.error("FFTLatentAttentionGatePooling not found in model. "
                      "Make sure pooling is set to 'fft_latent_attn_gate'.")
        return

    threshold = str(cfg.data.threshold)
    psd_cfg = cfg.get("psd_gate_analysis", {})
    k_bands = psd_cfg.get("k_bands", 8)
    base = psd_cfg.get("base", 4)
    band_mode = psd_cfg.get("band_mode", "uniform")
    splits = psd_cfg.get("splits", ["test"])
    manifest_path_str = str(getattr(cfg.get("analysis", {}), "peptide_manifest", "") or "").strip()
    if manifest_path_str and global_rank == 0:
        allowed_idx_log, _ = load_peptide_manifest(manifest_path_str)
        Logger.info(f"[Exp2] peptide manifest: {manifest_path_str} idx={sorted(allowed_idx_log)}")

    for condition in cfg.data[cfg.task.type].condition:
        for diff in cfg.data.diff:
            if cfg.data[cfg.task.type].mode == 'fix':
                if global_rank == 0:
                    Logger.info('loading datasets...')
                train_file_path = resolve_path(
                    cfg.data[cfg.task.type].fix.train_file,
                    diff=diff, condition=condition, threshold=threshold,
                    dataset=cfg.data[cfg.task.type].get('dataset'))
                valid_file_path = resolve_path(
                    cfg.data[cfg.task.type].fix.valid_file,
                    diff=diff, condition=condition, threshold=threshold,
                    dataset=cfg.data[cfg.task.type].get('dataset'))
                test_file_path = resolve_path(
                    cfg.data[cfg.task.type].fix.test_file,
                    diff=diff, condition=condition, threshold=threshold,
                    dataset=cfg.data[cfg.task.type].get('dataset'))

                train_loader = make_loader(local_rank, train_file_path, cfg=cfg, batch_size=cfg.train.batch_size,
                                           vocab_dict=vocab_dict, pin_memory=False,
                                           num_workers=cfg.train.num_workers, random_seed=random_seed)
                valid_loader = make_loader(local_rank, valid_file_path, cfg=cfg, batch_size=cfg.train.batch_size,
                                           vocab_dict=vocab_dict, pin_memory=False,
                                           num_workers=cfg.train.num_workers, random_seed=random_seed)
                test_loader = make_loader(local_rank, test_file_path, cfg=cfg, batch_size=cfg.train.batch_size,
                                          vocab_dict=vocab_dict, pin_memory=False,
                                          num_workers=cfg.train.num_workers, random_seed=random_seed)
                dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

                if global_rank == 0:
                    Logger.info('loading checkpoint...')
                if not cfg.model[cfg.task.type].check_point.load:
                    Logger.info('please give a check point path and set "load" to true')
                    return
                model = load_model(model, cfg.model[cfg.task.type].check_point.path, device)

                # 重新获取 gate_pooling (load_model 可能改变了 model)
                actual_model = model.module if isinstance(model, DDP) else model
                gate_pooling = find_fft_gate_pooling(actual_model)
                if gate_pooling is None:
                    Logger.error("FFTLatentAttentionGatePooling not found after loading checkpoint.")
                    return

                evaluator = Evaluator(model, dataloaders, metric_func, feature_fetcher, device, cfg)

                for split in splits:
                    if global_rank == 0:
                        Logger.info(f"===== Processing split: {split} =====")

                    gate_collector = GateWeightCollector()
                    gate_collector.register(gate_pooling)

                    allowed_idx, max_samples = get_analysis_filter(cfg)
                    seen = 0
                    per_sample_gate_frames: List[pd.DataFrame] = []
                    sample_idx_list: List[int] = []

                    loader = dataloaders[split]
                    model.eval()

                    with torch.no_grad():
                        for data in loader:
                            sequence, name2id, label = data
                            batch_idxs = _idx_list_from_name2id(name2id)
                            if allowed_idx is not None:
                                manifest_idxs = [i for i in batch_idxs if i in allowed_idx]
                                if not manifest_idxs:
                                    continue
                                if max_samples is not None and seen >= max_samples:
                                    break
                            elif max_samples is not None and seen >= max_samples:
                                break

                            token_sequence = feature_fetcher.query_features(sequence["peptide"])
                            batch1 = {
                                k: torch.tensor(v).to(device)
                                for k, v in token_sequence.items()
                            }
                            actual_model(batch1)
                            gate_collector.collect_from_module()

                            mod = gate_pooling
                            if hasattr(mod, "_last_freq_tokens") and hasattr(mod, "_last_enhanced_freq"):
                                ft = mod._last_freq_tokens.cpu()
                                ef = mod._last_enhanced_freq.cpu()
                                for bi, idx_val in enumerate(batch_idxs):
                                    if allowed_idx is not None and idx_val not in allowed_idx:
                                        continue
                                    ps_df = compute_per_sample_gate_by_band(
                                        ft[bi],
                                        ef[bi],
                                        idx=idx_val,
                                        k_bands=k_bands,
                                        base=base,
                                        band_mode=band_mode,
                                    )
                                    per_sample_gate_frames.append(ps_df)
                                    sample_idx_list.append(idx_val)
                                    seen += 1
                                    if max_samples is not None and seen >= max_samples:
                                        break

                    gate_data = gate_collector.aggregate()
                    gate_collector.remove()

                    if global_rank == 0:
                        Logger.info(
                            f"Collected gate data: {gate_data.get('raw_gate', torch.empty(0)).shape[0] if 'raw_gate' in gate_data else 0} samples; "
                            f"manifest idx={sorted(set(sample_idx_list))}"
                        )
                        if per_sample_gate_frames:
                            ps_gate = pd.concat(per_sample_gate_frames, ignore_index=True)
                            ps_gate.to_csv("./per_sample_gate_by_band.csv", index=False)
                            Logger.info(f"[saved] ./per_sample_gate_by_band.csv ({len(ps_gate)} rows)")

                    metrics = evaluator.run(split)

                    # ===================== 3) 获取 all_layer_features (从 evaluator) =====================
                    all_layer_features = evaluator.all_layer_features
                    if isinstance(all_layer_features, (list, tuple)):
                        if len(all_layer_features) > 0:
                            all_layer_features = torch.cat(all_layer_features, dim=0)
                        else:
                            all_layer_features = None
                    elif isinstance(all_layer_features, torch.Tensor):
                        if all_layer_features.numel() == 0:
                            all_layer_features = None
                    elif all_layer_features is not None:
                        Logger.warning(
                            f"Unexpected all_layer_features type: {type(all_layer_features)}"
                        )
                        all_layer_features = None
                    if all_layer_features is None:
                        Logger.warning("all_layer_features is empty, skipping PSD analysis")

                    if global_rank == 0:
                        # ===================== 4) PSD 频谱能量分析 =====================
                        Logger.info("=== PSD Spectral Energy Analysis ===")

                        if all_layer_features is not None:
                            N, T, D, L = all_layer_features.shape
                            Logger.info(f"all_layer_features shape: N={N}, T={T}, D={D}, L={L}")

                            # 沿 hidden_dim 做 FFT，分析每层频谱
                            # 取所有 token 的平均 -> (N, D, L)
                            x_for_fft = all_layer_features.mean(dim=1)  # (N, D, L)

                            stats = fft_layer_freq_distribution(
                                x_for_fft, use_rfft=True, measure="mag2", topk=3
                            )
                            prob_per_layer = stats['prob_per_layer']  # (L, F)

                            Logger.info(f"PSD computed: {L} layers, {int(stats['F'].item())} frequency bins")

                            # 绘制 PSD 分布图
                            plot_psd_band_distribution(
                                prob_per_layer,
                                k_bands=k_bands,
                                base=base,
                                band_mode=band_mode,
                                title=f"PSD Energy Distribution ({cfg.model[cfg.task.type].version}, {split})",
                                out_png=f"./psd_band_distribution_{split}.png",
                                out_csv=f"./psd_band_energy_{split}.csv",
                                highlight_band_range=(0, 3),
                            )

                            # 绘制累积曲线
                            plot_psd_cumulative_curve(
                                prob_per_layer,
                                k_bands=k_bands,
                                base=base,
                                band_mode=band_mode,
                                threshold=0.9,
                                title=f"Cumulative PSD Energy ({cfg.model[cfg.task.type].version}, {split})",
                                out_png=f"./psd_cumulative_curve_{split}.png",
                                highlight_band_range=(0, 3),
                            )

                            # 打印统计摘要
                            df_conc = compute_band_energy_concentration(
                                prob_per_layer, k_bands=k_bands, base=base, band_mode=band_mode
                            )
                            Logger.info("=== PSD Band Energy Concentration ===")
                            for _, row in df_conc.iterrows():
                                Logger.info(f"  Layer {int(row['layer'])}, Band {int(row['band'])}: "
                                             f"share={row['energy_share']:.4f}, cum={row['cumulative_share']:.4f}")
                        else:
                            Logger.warning("Skipping PSD analysis: no all_layer_features")

                        # ===================== 5) Gate 权重分析 =====================
                        Logger.info("=== Gate Weight Distribution Analysis ===")

                        if 'freq_tokens' in gate_data and len(gate_data['freq_tokens']) > 0:
                            seq_len = T if all_layer_features is not None else cfg.data.max_length
                            df_gate = plot_gate_weight_distribution(
                                gate_data,
                                seq_len=seq_len,
                                k_bands=k_bands,
                                base=base,
                                band_mode=band_mode,
                                threshold=0.9,
                                title_prefix=f"{cfg.model[cfg.task.type].version}",
                                out_dir=".",
                            )
                        else:
                            Logger.warning("No gate data collected, skipping gate analysis")

                # ===================== 6) 常规评估指标 =====================
                manifest_idx_final, _ = get_analysis_filter(cfg)
                eval_splits = (
                    ['test']
                    if manifest_idx_final is not None
                    else ['train', 'valid', 'test']
                )
                if global_rank == 0:
                    Logger.info(f"evaluating splits for metrics: {eval_splits}")
                for eval_split in eval_splits:
                    eval_metrics = evaluator.run(eval_split)
                    if global_rank == 0:
                        metric_func(evaluator.y_pred, evaluator.y_true, split=eval_split, plot=True)
                        Logger.info(f"{eval_split} metrics: {eval_metrics}")

    if cfg.mode.ddp:
        cleanup_multinodes()


if __name__ == '__main__':
    main()
    print('done.')

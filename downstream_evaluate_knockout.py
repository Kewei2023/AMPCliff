import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
import hydra
from pathlib import Path
from omegaconf import DictConfig
from AMPCliff.utils.std_logger import Logger
from AMPCliff.utils.scheduler import lr_scheduler
from AMPCliff.utils.distribution import setup_multinodes, cleanup_multinodes
from AMPCliff.loader.utils import make_loader #######
from AMPCliff.loader.split import random_split_data,stratified_split_data,fixed_cluster_split_data
from AMPCliff.utils.utils import get_device,fix_random_seed, load_weights, load_model
from AMPCliff.utils.path_helper import resolve_path
from AMPCliff.utils.evaluator import Evaluator #######
from AMPCliff.features.feature_fetcher import FeatureFetcher
from AMPCliff.factory.initializer import ModelInitializer
from AMPCliff.attention_knockout.as_ko import PostSoftmaxAttentionKnockout,PreSoftmaxAttentionKnockout
from AMPCliff.attention_knockout.hs_ko import HiddenStateMaskHook 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from AMPCliff.utils.metrics import Metrics
from AMPCliff.visualization.plot import plot_low_dimension
from AMPCliff.utils.match import seq2pair
import matplotlib.pyplot as plt
import mlflow
import torch
import json
from typing import Any, Dict, List, Optional, Set, Tuple

_DEBUG_LOG = Path(__file__).resolve().parent / ".cursor" / "debug-498d08.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        import time as _time

        payload = {
            "sessionId": "498d08",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(_time.time() * 1000),
        }
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG, "a", encoding="utf-8") as _df:
            _df.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


def _resolve_data_file(orig_cwd: str, template: str, **kwargs) -> str:
    """Resolve template placeholders and anchor relative paths to Hydra orig cwd."""
    resolved = resolve_path(template, **kwargs)
    if not os.path.isabs(resolved):
        resolved = os.path.join(orig_cwd, resolved)
    return os.path.normpath(resolved)


# ===== 工具函数 =====
def _predict_split(evaluator, split: str):
    """跑一次 evaluator 并返回 (y_pred, ids)，均为 1D torch.Tensor"""
    evaluator.run(split)
    y_pred, ids = evaluator.y_pred.view(-1), evaluator.ids.view(-1)
    return y_pred.detach().cpu(), ids.detach().cpu()

def _align_by_idx(base_ids, base_pred, ko_ids, ko_pred):
    """按 Idx 对齐 baseline 与 KO 的预测（外连接后仅保留交集顺序）。"""
    dfb = pd.DataFrame({"Idx": base_ids.numpy(), "pred_base": base_pred.numpy()})
    dfk = pd.DataFrame({"Idx": ko_ids.numpy(),   "pred_ko":   ko_pred.numpy()})
    merged = pd.merge(dfb, dfk, on="Idx", how="inner").sort_values("Idx")
    return (torch.from_numpy(merged["pred_base"].values).float(),
            torch.from_numpy(merged["pred_ko"].values).float())

def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.view(-1).float(); b = b.view(-1).float()
    return torch.mean((a - b) ** 2).item()

def _infer_seq_len(data, tokenizer,feature_fetcher, device) -> int:
    # for data in dataloader:
    sequence, name2id, label = data
            
    token_sequence = feature_fetcher.query_features(sequence['peptide'])
    
    batch = {k: torch.tensor(v).to(device) for k, v in token_sequence.items()}
    # ipdb.set_trace()
            
    if "attention_mask" in batch:
        return batch["attention_mask"].sum(-1)[0].item(),sequence
        
    return 0,sequence

def _idx_from_data(name2id) -> int:
    """Extract sample Idx from collated name2id (seqName -> Idx list)."""
    for _name, ids in name2id.items():
        if isinstance(ids, (list, tuple)):
            return int(ids[0])
        return int(ids)
    raise ValueError("empty name2id in batch")


def _rel_pos(token_pos: int, seq_len: int) -> float:
    denom = max(seq_len - 3, 1)
    return float(token_pos - 1) / float(denom)


def _load_peptide_manifest(path: str) -> Tuple[Set[int], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    idx_list = {int(x) for x in payload["idx_list"]}
    return idx_list, payload


def _resolve_layer_ids(cfg, num_layers: int) -> List[int]:
    if bool(getattr(cfg.knockout, "last_layer_only", False)):
        return [num_layers - 1]
    layer_ids = list(getattr(cfg.knockout, "layer_ids", []))
    if layer_ids:
        return layer_ids
    return list(range(num_layers))


def _predict_one(model, feature_fetcher, data, device, cfg) -> float:
    """Single-peptide forward; returns scalar prediction."""
    sequence, name2id, label = data
    token_sequence = feature_fetcher.query_features(sequence["peptide"])
    batch = {k: torch.tensor(v).to(device) for k, v in token_sequence.items()}
    true_labels = torch.tensor(label[cfg.task.type]).to(device)
    model.eval()
    with torch.no_grad():
        if cfg.mode.amp:
            with torch.cuda.amp.autocast():
                output = model(batch)
        else:
            output = model(batch)
    return float(output[0].squeeze().cpu().item())


def _knockout_output_path(cfg, default_name: str = "knockout_lastlayer_HS.csv") -> Path:
    out_dir = str(getattr(cfg.knockout, "output_dir", "") or "").strip()
    if out_dir:
        p = Path(out_dir)
    else:
        p = Path(".")
    p.mkdir(parents=True, exist_ok=True)
    return p / default_name


def _knockout_once(encoder, mode: str, layer_id: int, query_token_pos_list: int,key_token_pos_list,renorm: bool):
    """注册一次 KO，返回 hook 实例，记得外部调用后 .remove()"""
    if mode.upper() == "HS":
        hook = HiddenStateMaskHook(encoder, [layer_id], token_positions=query_token_pos_list)
    elif mode.upper() == "AS":
        # hook = PostSoftmaxAttentionKnockout(encoder, [layer_id],
        #                                     row_indices=query_token_pos_list,
        #                                     col_indices=key_token_pos_list,
        #                                     renorm=renorm)
        hook = PreSoftmaxAttentionKnockout(encoder, [layer_id],
                                            # dtype=dtype,
                                            # device=device,
                                            row_indices=query_token_pos_list,
                                            col_indices=key_token_pos_list,
                                            )
    else:
        raise ValueError("cfg.knockout.mode must be 'AS' or 'HS'")
    hook.register()
    return hook
########################################################################
# util

@hydra.main(config_path="configs", config_name="downstream_knockout.yaml")
def main(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    cfg.orig_cwd = orig_cwd
    local_rank = 0
    global_rank = 0

    assert cfg.train.batch_size == 1, "batch size should set to 1 for knockout analysis"
    # ipdb.set_trace()
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
        # deepspeed.init_distributed()
    else:
        random_seed = cfg.train.random_seed
        device = get_device(cfg)

    if global_rank == 0:
        Logger.info("setting random seed: {}".format(random_seed))

    fix_random_seed(random_seed, cuda_deterministic=True)
    
    if cfg.logger.log and global_rank == 0:

        # log hyper-parameters
        for p, v in cfg.mode.items():
            mlflow.log_param(p, v)
            
        for p, v in cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].items():
           
            try:
              mlflow.log_param(p, v)
            except:
              mlflow.log_param(p, v[:500])

        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        for p, v in cfg.model[cfg.task.type].items():
           mlflow.log_param(p, v)



    metric_func = Metrics(cfg.task.type,topK=50)
    model, tokenizer = ModelInitializer(cfg,device).init()
    
    vocab_dict = tokenizer.get_vocab()
    
    feature_fetcher = FeatureFetcher(cfg,tokenizer)
    
    if cfg.mode.ddp:
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True
                    )

   
    threshold = str(cfg.data.threshold)
    for condition in cfg.data[cfg.task.type].condition:
          
      for diff in cfg.data.diff: 
      
        if cfg.data[cfg.task.type].mode == 'fix':
    
            # diff = cfg.data.diff
            if global_rank == 0:
                Logger.info('loading train data...')
            
            train_file_path = _resolve_data_file(
                orig_cwd,
                cfg.data[cfg.task.type].fix.train_file,
                diff=diff,
                condition=condition,
                threshold=threshold,
                dataset=cfg.data[cfg.task.type].get('dataset')
            )
            # #region agent log
            _debug_log(
                "H1",
                "downstream_evaluate_knockout.py:train_path",
                "resolved train csv",
                {
                    "orig_cwd": orig_cwd,
                    "hydra_cwd": os.getcwd(),
                    "train_file_path": train_file_path,
                    "exists": os.path.isfile(train_file_path),
                },
            )
            # #endregion

            train_dataloader = make_loader(
                                            local_rank=local_rank,
                                            dataset_file=train_file_path,
                                            cfg = cfg,
                                            batch_size=cfg.train.batch_size,
                                            vocab_dict=vocab_dict,
                                            pin_memory=False,
                                            num_workers=cfg.train.num_workers,
                                            random_seed=random_seed
                                        )
            if global_rank == 0:
                Logger.info('loading valid data...')

            valid_file_path = _resolve_data_file(
                orig_cwd,
                cfg.data[cfg.task.type].fix.valid_file,
                diff=diff,
                condition=condition,
                threshold=threshold,
                dataset=cfg.data[cfg.task.type].get('dataset')
            )
              
            
            valid_dataloader = make_loader(
                                            local_rank=local_rank,
                                            dataset_file=valid_file_path,
                                            cfg = cfg,
                                            batch_size=cfg.train.batch_size,
                                            vocab_dict=vocab_dict,
                                            pin_memory=False,
                                            num_workers=cfg.train.num_workers,
                                            random_seed=random_seed
                                        )
            if global_rank == 0:
                Logger.info('loading test data...')

            test_file_path = _resolve_data_file(
                orig_cwd,
                cfg.data[cfg.task.type].fix.test_file,
                diff=diff,
                condition=condition,
                threshold=threshold,
                dataset=cfg.data[cfg.task.type].get('dataset')
            )
              
            
            test_dataloader = make_loader(
                                            local_rank=local_rank,
                                            dataset_file=test_file_path,
                                            cfg = cfg,
                                            batch_size=cfg.train.batch_size,
                                            vocab_dict=vocab_dict,
                                            pin_memory=False,
                                            num_workers=cfg.train.num_workers,
                                            random_seed=random_seed
                                        )
    
            dataloaders = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}
            
            if global_rank == 0:
                Logger.info('loading checkpoint...')
                
            if not cfg.model[cfg.task.type].check_point.load and not cfg.model[cfg.task.type].check_point.pretrain_only :
              
              Logger.info('please give a check point path and set "load" to true or set pretrain_only to true')
              exit()
              
            elif cfg.model[cfg.task.type].check_point.load:
              model = load_model(model, cfg.model[cfg.task.type].check_point.path, device)
           
                
            if global_rank == 0 and getattr(cfg, "knockout", None) and cfg.knockout.enabled:
                Logger.info(
                    f"[Knockout] mode={cfg.knockout.mode}, split={getattr(cfg.knockout, 'split', 'test')}, "
                    f"last_layer_only={getattr(cfg.knockout, 'last_layer_only', False)}"
                )

                KO_SPLIT = str(getattr(cfg.knockout, "split", "test")).lower()
                if KO_SPLIT == "train":
                    KO_loader = dataloaders["train"]
                elif KO_SPLIT == "valid":
                    KO_loader = dataloaders["valid"]
                else:
                    KO_loader = dataloaders["test"]

                allowed_idx: Optional[Set[int]] = None
                manifest_path = str(getattr(cfg.knockout, "peptide_manifest", "") or "").strip()
                if manifest_path:
                    allowed_idx, manifest_meta = _load_peptide_manifest(manifest_path)
                    Logger.info(f"[Knockout] peptide manifest: {manifest_path} idx={sorted(allowed_idx)}")

                encoder = model.module.pretrain_model.encoder if isinstance(model, DDP) else model.pretrain_model.encoder
                num_layers = len(encoder.layer)
                layer_ids = _resolve_layer_ids(cfg, num_layers)
                max_samples = getattr(cfg.knockout, "max_samples", None)
                save_heatmap = bool(getattr(cfg.knockout, "save_per_peptide_heatmap", False))

                pooling_tag = str(cfg.model[cfg.task.type].pooling)
                dataset_tag = str(cfg.data[cfg.task.type].get("dataset", ""))
                model_version = str(cfg.model[cfg.task.type].version)
                train_seed = int(cfg.train.random_seed)

                long_rows: List[dict] = []
                seen = 0

                for _loader_idx, data in enumerate(KO_loader):
                    sample_idx = _idx_from_data(data[1])
                    if allowed_idx is not None and sample_idx not in allowed_idx:
                        continue
                    if max_samples is not None and seen >= max_samples:
                        Logger.info(f"[Knockout] reached max_samples={max_samples}")
                        break
                    seen += 1

                    seq_len, sequence = _infer_seq_len(data, tokenizer, feature_fetcher, device)
                    peptide_seq = sequence["peptide"][0]
                    token_positions = list(range(1, seq_len - 1))
                    if not token_positions:
                        continue

                    pred_base = _predict_one(model, feature_fetcher, data, device, cfg)
                    layer_id = layer_ids[0]

                    diffs_1d = np.zeros(len(token_positions), dtype=np.float32)
                    for j, pos in enumerate(token_positions):
                        hook = _knockout_once(
                            encoder,
                            cfg.knockout.mode,
                            layer_id,
                            [pos],
                            token_positions,
                            cfg.knockout.renorm,
                        )
                        try:
                            pred_ko = _predict_one(model, feature_fetcher, data, device, cfg)
                        finally:
                            hook.remove()
                        abs_delta = abs(pred_ko - pred_base)
                        diffs_1d[j] = abs_delta
                        long_rows.append(
                            {
                                "idx": sample_idx,
                                "peptide": peptide_seq,
                                "seq_len": int(seq_len),
                                "token_pos": int(pos),
                                "rel_pos": _rel_pos(pos, seq_len),
                                "abs_delta": float(abs_delta),
                                "pred_base": float(pred_base),
                                "pred_ko": float(pred_ko),
                                "layer": int(layer_id),
                                "pooling": pooling_tag,
                                "model_version": model_version,
                                "dataset": dataset_tag,
                                "train_seed": train_seed,
                                "split": KO_SPLIT,
                            }
                        )

                    if save_heatmap and len(layer_ids) == 1:
                        out_prefix = getattr(
                            cfg.knockout,
                            "out_prefix",
                            f"{model_version}-{condition}-diff{diff}-{KO_SPLIT}",
                        )
                        heat_png = (
                            f"./{out_prefix}-KO_{cfg.knockout.mode}_idx{sample_idx}.png"
                        )
                        plt.figure(figsize=(1.2 * len(token_positions) + 2, 3))
                        plt.plot(token_positions, diffs_1d, marker="o", markersize=3)
                        plt.xlabel("Token position")
                        plt.ylabel("|Δ prediction|")
                        plt.title(f"idx={sample_idx} layer={layer_id} pooling={pooling_tag}")
                        plt.tight_layout()
                        plt.savefig(heat_png, dpi=150)
                        plt.close()
                        Logger.info(f"[Knockout] saved heatmap: {heat_png}")

                csv_path = _knockout_output_path(cfg)
                pd.DataFrame(long_rows).to_csv(csv_path, index=False)
                Logger.info(f"[Knockout] saved {len(long_rows)} rows -> {csv_path}")

    if cfg.mode.ddp:
        cleanup_multinodes()
      
if __name__ == '__main__':
    main()
    print('done.')

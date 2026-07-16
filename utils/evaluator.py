# maintained by kewei li

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import ipdb
from torch.nn import MSELoss
from itertools import chain
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from AMPCliff.utils.fftlag_manifest_utils import (
    get_analysis_filter,
    idx_list_from_name2id,
)


class Evaluator():

    def __init__(self, net,dataloaders,metric_func,feature_fetcher, device, cfg):
        self.net = net
        self.feature_fetcher = feature_fetcher
        self.device = device
        # self.criterion = criterion
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.metrics_func = metric_func
        if cfg.train.loss =='mse':
            self.reg_loss_func = MSELoss()
        self.scaler = GradScaler()
        
    def run(self,split):
        self.net.eval()

        
        with torch.no_grad():
            total_loss = []
            loss = 0
            y_pred, y_true = [],[]
            names, ids = [], []
            latent = []
            latent_all_layers = []
            metrics = {}
            allowed_idx, max_samples = get_analysis_filter(self.cfg)
            if allowed_idx is not None and max_samples is None:
                max_samples = len(allowed_idx)
            seen = 0
            collected_idx = set()
            n_batches = 0
            n_skipped_manifest = 0
            for step, data in tqdm(
                    enumerate(self.dataloaders[split]),
                    total=len(self.dataloaders[split]),
                    desc="evaluating | loss: {:.4f}".format(loss)):

                sequence, name2id, label = data
                n_batches += 1
                batch_idxs = idx_list_from_name2id(name2id)

                if allowed_idx is not None:
                    manifest_idxs = [i for i in batch_idxs if i in allowed_idx]
                    if not manifest_idxs:
                        n_skipped_manifest += 1
                        continue
                    if all(i in collected_idx for i in manifest_idxs):
                        n_skipped_manifest += 1
                        continue
                    collected_idx.update(manifest_idxs)
                if max_samples is not None and seen >= max_samples:
                    break
                seen += 1

                token_sequence = self.feature_fetcher.query_features(sequence['peptide'])
                
                batch = {k: torch.tensor(v).to(self.device) for k, v in token_sequence.items()}
                
                noised_labels = torch.tensor(label[f'noised_{self.cfg.task.type}']).to(self.device)
                true_labels = torch.tensor(label[self.cfg.task.type]).to(self.device)
                
                if self.cfg.mode.amp:
                    with autocast():
                        output = self.net(batch)
                        loss = self.reg_loss_func(output[0].squeeze(),true_labels.squeeze())

                else:
                    output = self.net(batch)
                    loss = self.reg_loss_func(output[0].squeeze(),true_labels.squeeze())

                

                y_pred.append(output[0].cpu())
                y_true.append(true_labels.cpu())
                names.extend(list(name2id.keys()))
                ids.extend(list(chain.from_iterable(list(name2id.values()))))
                latent.append(output[1].cpu())
                latent_all_layers.append(output[2].cpu())
                if allowed_idx is not None and len(collected_idx) >= len(allowed_idx):
                    break
                # ipdb.set_trace()
            if not y_pred:
                raise RuntimeError(
                    f"Evaluator.run({split}): no samples collected "
                    f"(batches={n_batches}, skipped_manifest={n_skipped_manifest}, "
                    f"allowed_idx={sorted(allowed_idx) if allowed_idx else None}). "
                    "Check peptide manifest vs dataloader shuffle."
                )
            y_pred = torch.cat(y_pred)
            y_true = torch.cat(y_true)
            latent = torch.cat(latent).mean(1)
            
            all_layer_features = torch.cat(latent_all_layers)
            latent_all_layers = all_layer_features.mean(1)

            names = torch.tensor(names)
            ids = torch.tensor(ids)
            # total_loss = np.array(total_loss)

        metrics = self.metrics_func(y_pred, y_true,split)
        res_dict = {}
        # for k,v in loss_dict.items():
        #     res_dict["{}_{}".format(split, k )] = np.mean(v)
        # res_dict['loss'] = np.mean(total_loss)
        for k, v in metrics.items():
            # res_dict[k] = v
            res_dict[f"{split}-{k}"] = np.mean(v)
        
        self.y_pred = y_pred
        self.y_true = y_true
        self.names = names
        self.ids = ids
        self.latent = latent # last hidden_states, seq_len dim by mean
        self.latent_all_layers = latent_all_layers  # all layer hidden_states by mean
        self.all_layer_features = all_layer_features  # all layer hidden_states originally (with seq_sim)
        return res_dict # ,y_pred, y_true, names, ids, latent


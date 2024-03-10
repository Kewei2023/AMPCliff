
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import ipdb
from torch.nn import MSELoss, SmoothL1Loss
from itertools import chain
from .loss import TolerantMSELossv2



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
        if cfg.train.loss =='huber':
            self.reg_loss_func = SmoothL1Loss(beta=cfg.train.loss_beta)
        if cfg.train.loss =='tmse':
            self.reg_loss_func = TolerantMSELossv2(margin=cfg.train.loss_beta)
        
    def run(self,split):
        self.net.eval()

        
        with torch.no_grad():
            total_loss = []
            loss = 0
            y_pred, y_true = [],[]
            names, ids = [], []
            latent = []
            metrics = {}
            for step, data in tqdm(
                    enumerate(self.dataloaders[split]),
                    total=len(self.dataloaders[split]),
                    desc="evaluating | loss: {:.4f}".format(loss)):

                sequence, name2id, label = data
                
                token_sequence = self.feature_fetcher.query_features(sequence['peptide'])
                
                batch = {k: torch.tensor(v).to(self.device) for k, v in token_sequence.items()}
                
                noised_labels = torch.tensor(label[f'noised_{self.cfg.task.type}']).to(self.device)
                true_labels = torch.tensor(label[self.cfg.task.type]).to(self.device)
                
                output = self.net(batch)
                
                loss = self.reg_loss_func(output[0].squeeze(),noised_labels.squeeze())


                y_pred.append(output[0].cpu())
                y_true.append(true_labels.cpu())
                names.extend(list(name2id.keys()))
                ids.extend(list(chain.from_iterable(list(name2id.values()))))
                latent.append(output[1].cpu())
                # ipdb.set_trace()
                # y_pred = output.logits
                # y_true = batch['labels']
                # batch_metrics = self.metrics_func(y_pred, y_true,split)

                # for k, v in batch_metrics.items():
                #     if k not in metrics:
                #         metrics[k] = [v]
                #     else:
                #         metrics[k].append(v)
                # total_loss.append(loss)
                # ipdb.set_trace()
                # mask_pos.extend(label['pos'])
            y_pred = torch.cat(y_pred)
            y_true = torch.cat(y_true)
            latent = torch.cat(latent).mean(1)
            
            # names = torch.cat(names)
            # ids = torch.cat(ids)
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
        self.latent = latent
        return res_dict # ,y_pred, y_true, names, ids, latent


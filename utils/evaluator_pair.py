
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import ipdb
from itertools import chain
from torch.nn import MSELoss, SmoothL1Loss


class Evaluator():

    def __init__(self, net,dataloaders,metric_func,feature_fetcher, device, cfg):
        self.net = net
        self.feature_fetcher = feature_fetcher
        self.device = device
        # self.criterion = criterion
        self.reg_loss_func = MSELoss()
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.metrics_func = metric_func

    def run(self,split):
        self.net.eval()

        
        with torch.no_grad():
            total_loss = []
            loss = 0
            y_pred_1, y_true_1 = [],[]
            y_pred_2, y_true_2 = [],[]
            y_pred_12, y_true_12 = [],[]
            pairIDs = []
            for step, data in tqdm(
                    enumerate(self.dataloaders[split]),
                    total=len(self.dataloaders[split]),
                    desc="evaluating | loss: {:.4f}".format(loss)):

                sequence1, sequence2, pairids, label= data
                # ipdb.set_trace()
                token_sequence1 = self.feature_fetcher.query_features(sequence1['peptide'])
                token_sequence2 = self.feature_fetcher.query_features(sequence2['peptide'])
                batch1 = {k: torch.tensor(v).to(self.device) for k, v in token_sequence1.items()}
                   
                batch2 = {k: torch.tensor(v).to(self.device) for k, v in token_sequence2.items()}
                
                regression_output1, regression_output2,regression_output12 = self.net(batch1,batch2)
                # ipdb.set_trace()
                # rank_loss = self.rank_loss_func(rank_output1, rank_output2)
                reg_loss1 = self.reg_loss_func(regression_output1.squeeze().cpu(),torch.tensor(label['regression1']).squeeze())
                reg_loss2 = self.reg_loss_func(regression_output2.squeeze().cpu(),torch.tensor(label['regression2']).squeeze())
                reg_loss12 = self.reg_loss_func(regression_output12.squeeze().cpu(),torch.tensor(label['regression12']).squeeze())
                
                if self.cfg.train.loss_type == '12+1+2':
                    loss = reg_loss12 + reg_loss1 + reg_loss2
                if self.cfg.train.loss_type == '1+2':   
                    loss = reg_loss1 + reg_loss2
                if self.cfg.train.loss_type == '12':   
                    loss = reg_loss12

                y_pred_1.append(regression_output1)
                y_true_1.append(torch.tensor(label['regression1']))

                y_pred_2.append(regression_output2)
                y_true_2.append(torch.tensor(label['regression2']))

                y_pred_12.append(regression_output12)
                y_true_12.append(torch.tensor(label['regression12']))

                pairIDs.extend(pairids['pairID'])
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
            y_pred_1 = torch.cat(y_pred_1)
            y_true_1 = torch.cat(y_true_1)
            y_pred_2 = torch.cat(y_pred_2)
            y_true_2 = torch.cat(y_true_2)
            y_pred_12 = torch.cat(y_pred_12)
            y_true_12 = torch.cat(y_true_12)

            y_pred = [y_pred_1,y_pred_2]
            y_true = [y_true_1,y_true_2]
            # IDs = [f'{p}-p0' for p in pairIDs] + [f'{p}-p1' for p in pairIDs]
            
        metrics1 = self.metrics_func(y_pred_1, y_true_1,split)
        metrics2 = self.metrics_func(y_pred_2, y_true_2,split)
        metrics12 = self.metrics_func(y_pred_12, y_true_12,split)
        
        
        res_dict = {}
        # for k,v in loss_dict.items():
        #     res_dict["{}_{}".format(split, k )] = np.mean(v)
        # res_dict['loss'] = np.mean(total_loss)
        for k in metrics1:
            res_dict[f"{split}-{k}-1"] = metrics1[k]
            res_dict[f"{split}-{k}-2"] = metrics2[k]
            res_dict[f"{split}-{k}-12"] = metrics12[k]
            
        
        self.y_pred_1 = y_pred_1
        self.y_true_1 = y_true_1
        self.y_pred_2 = y_pred_2
        self.y_true_2 = y_true_2
        self.y_pred_12 = y_pred_12
        self.y_true_12 = y_true_12
        self.ids = pairIDs

        self.y_pred = y_pred
        self.y_true = y_true
        # self.ids = IDs

        return res_dict


import traceback
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import ipdb
def cal_recall(y_pred, y_true, top):
    a_sort_idx = y_pred.argsort()
    b_sort_idx = y_true.argsort()
    
    recall = len(set(b_sort_idx[-top:].tolist()).intersection(a_sort_idx[-top:].tolist()))

    return recall


class MulticlassMetrics():
    def __init__(self):
        pass
    
    def __call__(self, pred, true):
        
        # ipdb.set_trace()
        y_true = true.cpu()

        masked_ids = (y_true+1).nonzero()

        y_true = y_true[masked_ids[:,0],masked_ids[:,1]]

        true = true.cpu().view(-1)

        y_pred = torch.argmax(torch.softmax(pred, dim=-1), dim = -1).cpu()
        
        y_pred = y_pred[masked_ids[:,0],masked_ids[:,1]]
        
        pred = pred.cpu().view(true.shape[0],-1)
        
        # # .numpy()
        # .numpy()

        res_dict = {}
        # ipdb.set_trace()
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        # total_mask_loss, total_loss = [], []
        # total_loss.append(loss_fct(pred,true).item())
        # res_dict['mask_loss'] = torch.mean(torch.tensor(total_mask_loss))
        res_dict['loss'] = loss_fct(pred,true).item()

        
        res_dict['accuracy'] = accuracy_score(y_pred,y_true)

        return res_dict

class BinaryMetrics():
    def __init__(self):
        pass
    
    def __call__(self, pred, true):
        
        
        y_pred = torch.round(pred).cpu().numpy()
        y_score=pred.squeeze(1).cpu().numpy()
        true = true.cpu().numpy()

        res_dict = {}

        report = classification_report(true, y_pred)
        
        pre_score = precision_score(true, y_pred, average="macro")
        rec_score = recall_score(true, y_pred, average="macro")
        try:
            auc_score = roc_auc_score(true, y_score, average="macro")
        except:
            traceback.print_exc()
            auc_score = 0
            
        res_dict["auc_score"] = auc_score
        res_dict["precision_score"] = pre_score
        res_dict["recall_score"] = rec_score
        res_dict["classification_report"] = report

        return res_dict
            
class AffinityMetrics():
    def __init__(self, topK=10):
        self.topK = topK
        
    def __call__(self, pred, true, split,plot=False):
        
        if isinstance(true, torch.Tensor):
          true = true.cpu().numpy()
        elif isinstance(true, np.ndarray):
          pass
          
        if isinstance(pred, torch.Tensor):
          pred = pred.squeeze(1).cpu().numpy()
        elif isinstance(pred, np.ndarray):
          pass
          
        idxs = ~np.isnan(true)
        true = true[idxs]
        pred = pred[idxs]

        # Get the predictions
        if true.shape[0] > 0:
            mse = np.square(pred - true)
            self.mse = mse
            average_mse = np.mean(mse)
            # ipdb.set_trace()
            pearson_corr = pearsonr(pred, true)[0]
            spearman_corr = spearmanr(pred, true)[0]

            recall = cal_recall(pred, true, self.topK)
            if plot:
                df = pd.DataFrame(data = [[a, b] for a,b in zip(true, pred)], columns = ["original_MIC", "predicted_MIC"])
                sns.lmplot(x="original_MIC", y="predicted_MIC", data=df).set(title='{} set | total point: {}| spearman: {} | recall: {}'.format(split, df.shape[0], np.round(spearman_corr, 3),recall ))
                plt.savefig('{}_mic_corr_{}.png'.format(split, str(int(time.time()))), dpi=300, bbox_inches='tight')  
        else:
            pearson_corr = spearman_corr = recall = 0.0
        return {"mse": average_mse, "pearson": pearson_corr, "spearman": spearman_corr, "recall": recall}

class ContactmapMetrics():
    def __init__(self, contact_topK=100):
        self.k = contact_topK
        
    def __call__(self, pred, true, split): 

        auc_list = []
        all_zero = 0
        accs = []
        topk_recall_list = []
        for i in range(pred.shape[0]):
            indx = ~torch.isnan(true[i])
            
            try:
                auc_score = roc_auc_score(true[i][indx].detach().cpu().reshape(-1), pred[i][indx].detach().cpu().reshape(-1), average="macro")
                # print(auc_score)
                auc_list.append(auc_score)
                
                values, indices = torch.topk(pred[i][indx].view(-1), self.k)
                topk_recall = torch.sum(true[i][indx].view(-1)[indices])
                topk_recall_list.append(topk_recall.item())
            except:
                auc_list.append(0.0)
                topk_recall_list.append(0.0)
                all_zero += 1
            
        
            # pred_label = pred[i][indx].clone().detach().reshape(-1)
            # true_label = true[i][indx].clone().detach().reshape(-1)
            # pred_label[pred_label>=0.5] = 1
            # pred_label[pred_label<0.5] = 0
            # if len(pred_label) == 0:
            #     accuracy = 0
            # else:
            #     accuracy = torch.eq(pred_label, true_label).sum().float().item() / len(pred_label)
            # accs.append(accuracy)
            
        # precision_list, recall_list = [], []
        # for i in range(pred.shape[0]):
        #     indx = ~torch.isnan(true[i])
        #     try:
        #         precision = precision_score(true[i][indx].detach().cpu().reshape(-1), pred[i][indx].detach().cpu().reshape(-1))
        #         recall = recall_score(true[i][indx].detach().cpu().reshape(-1), pred[i][indx].detach().cpu().reshape(-1))
        #         precision_list.append(precision)
        #         recall_list.append(recall)
        #     except:
        #         continue
        self.auc_list = auc_list
        self.topk_recall_list = topk_recall_list
        
        auc_list = np.array(auc_list)
        topk_recall_list = np.array(topk_recall_list)
        
        indx = auc_list != 0
        average_auc = np.mean(auc_list[indx])
        average_topk_recall = np.mean(topk_recall_list[indx])
        
        plt.figure()
        sns.histplot(auc_list[indx])
        plt.axvline(x=average_auc, color="grey", linestyle='--')
        plt.xlabel('Average AUC')
        plt.ylabel('counts')
        plt.savefig('{}_average_auc_{}.png'.format(split, str(int(time.time()))), dpi=300, bbox_inches='tight')  
        
        # average_precision = np.mean(precision_list)
        # average_recall = np.mean(recall_list)
        # return {"average_auc": average_auc,
        #         "average_precision": average_precision,
        #         "average_recall": average_recall}
        return {"average_auc": average_auc, "topk_recall": average_topk_recall}
    
class Metrics():
    def __init__(self,task, topK=10, contact_topK=100):
        
        self.affinity_metrics = AffinityMetrics(topK)
        self.multiclass_metrics = MulticlassMetrics()
        # self.contactmap_metrics = ContactmapMetrics(contact_topK)
        self.binary_metrics = BinaryMetrics()
        self.task = task
    def __call__(self, pred, true,split=None,plot=False):

        
        res = {}

        if self.task == 'pretrain':
            metrics = self.multiclass_metrics(pred, true)
        if self.task == 'regression' or self.task == 'rank':
            metrics = self.affinity_metrics(pred, true, split,plot)
        res.update(metrics)
                
        return res
            
        
            
            
                
                
                
            

                
    
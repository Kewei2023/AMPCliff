import pandas as pd
from scipy.stats import pearsonr, spearmanr
import os

def cal_recall(y_pred, y_true, top):
    a_sort_idx = y_pred.argsort()
    b_sort_idx = y_true.argsort()
    
    recall = len(set(b_sort_idx[-top:].tolist()).intersection(a_sort_idx[-top:].tolist()))

    return recall


if __name__=='__main__':
  condition = 'tanimoto average'
  
  result_corr = {}
  print(f'condition is {condition}')
  if condition == 'blosum62 average':
    path = "/data/home/scv6872/AMPCliff/outputs/2024-03-31/16-44-07/"
    
  if condition == 'tanimoto average':
    path = "/data/home/scv6872/AMPCliff/outputs/2024-03-31/16-16-54/"
    
  for diff in [2,3,4,5]:
    results = pd.read_csv(os.path.join(path, f'diff{diff}','test_result.csv'))
    
    models_pred = [_ for _ in results.columns if 'pred' in _]
    
    true = results['Activity']
    
    result_corr[f'diff{diff}'] = {}
    
    for mdl_pred in models_pred:
      pearson_corr = pearsonr(results[mdl_pred], true)[0]
      spearman_corr = spearmanr(results[mdl_pred], true)[0]
      recall = cal_recall(results[mdl_pred], true, 50)
      result_corr[f'diff{diff}'][f'spearman_{mdl_pred.split("-")[0]}'] = spearman_corr
      result_corr[f'diff{diff}'][f'pearson_{mdl_pred.split("-")[0]}'] = pearson_corr
      result_corr[f'diff{diff}'][f'recall_{mdl_pred.split("-")[0]}'] = recall
      
      
  pd.DataFrame(result_corr).to_csv(os.path.join(path,'test_corr.csv'))
      
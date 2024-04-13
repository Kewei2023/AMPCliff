import pandas as pd
from scipy.stats import pearsonr, spearmanr
import os

def cal_recall(y_pred, y_true, top):
    a_sort_idx = y_pred.argsort()
    b_sort_idx = y_true.argsort()
    
    recall = len(set(b_sort_idx[-top:].tolist()).intersection(a_sort_idx[-top:].tolist()))

    return recall


if __name__=='__main__':
  # feature_name = 'AMPSpace' # 'AMPSpace', 'CellFree-cnn', 'CellFree-rnn', 'peptimizer'
  
  
  
  save_path = "/data/home/scv6872/AMPCliff/outputs/2024-04-04/"
  
  for condition in ['blosum62 average','tanimoto average']:
    result_corr = {}
    
    for diff in [2,3,4,5]:
      
      result_corr[f'diff{diff}'] = {}
      for feature_name in ['bert-base','esm2_t6','esm2_t12','esm2_t33','gpt2-base','protgpt2','progen2-small','progen2-base','progen2-medium']: # ['AMPSpace', 'CellFree-cnn', 'CellFree-rnn', 'peptimizer']:
        print(f'feature_name is {feature_name}')
        
        if feature_name == 'AMPSpace':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-01/22-26-35/"
          
        if feature_name == 'CellFree-cnn':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-01/22-29-01/"
        
        if feature_name == 'CellFree-rnn':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-01/22-36-13/"
          
        if feature_name == 'peptimizer':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-01/22-35-37/"
        
        if feature_name == 'bert-base':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-03/12-43-47/"
          
        if feature_name == 'esm2_t6':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-02/18-22-30/"
        
        if feature_name == 'esm2_t12':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-02/18-20-26/"
          
        if feature_name == 'esm2_t33':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-03/13-16-23/"
        
        if feature_name == 'gpt2-base':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-03/12-56-45/"
          
        if feature_name == 'protgpt2':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-03/13-03-43/"
          
        if feature_name == 'progen2-small':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-02/18-18-06/"
        
        if feature_name == 'progen2-base':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-03/12-50-48/"
          
        if feature_name == 'progen2-medium':
          path = "/data/home/scv6872/AMPCliff/outputs/2024-04-03/12-52-43/"
      
        results = pd.read_csv(os.path.join(path, f'{feature_name}-{condition}-diff{diff}-test_result.csv'))
        
        models_pred = [_ for _ in results.columns if 'pred' in _]
        
        true = results['Activity']
         
        for mdl_pred in models_pred:
          pearson_corr = pearsonr(results[mdl_pred], true)[0]
          spearman_corr = spearmanr(results[mdl_pred], true)[0]
          recall = cal_recall(results[mdl_pred], true, 50)
          result_corr[f'diff{diff}'][f'spearman_{mdl_pred.replace("-pred","")}'] = spearman_corr
          result_corr[f'diff{diff}'][f'pearson_{mdl_pred.replace("-pred","")}'] = pearson_corr
          result_corr[f'diff{diff}'][f'recall_{mdl_pred.replace("-pred","")}'] = recall
        
      
    pd.DataFrame(result_corr).to_csv(os.path.join(save_path,f'{condition}_all_test_corr.csv'))
      
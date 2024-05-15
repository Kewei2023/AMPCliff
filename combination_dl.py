import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score

def combination(ac_name,ramdon_name='pred'):
    file_path_ramdon = 'E:\\amp_c\\ramdon_split\\fold_5\\'+ac_name+'_final_results.csv'
    file_path_diff2 = 'E:\\amp_c\\AC_Splits\\blosum62_average\\'+ac_name+r'\\'+ac_name+'-blosum62 average-diff2-valid_result.csv'
    file_path_diff3 = 'E:\\amp_c\\AC_Splits\\blosum62_average\\' + ac_name + r'\\' + ac_name + '-blosum62 average-diff3-valid_result.csv'
    file_path_diff4 = 'E:\\amp_c\\AC_Splits\\blosum62_average\\' + ac_name + r'\\' + ac_name + '-blosum62 average-diff4-valid_result.csv'
    file_path_diff5 = 'E:\\amp_c\\AC_Splits\\blosum62_average\\' + ac_name + r'\\' + ac_name + '-blosum62 average-diff5-valid_result.csv'

    df1 = pd.read_csv(file_path_ramdon)
    df2 = pd.read_csv(file_path_diff2)
    df3 = pd.read_csv(file_path_diff3)
    df4 = pd.read_csv(file_path_diff4)
    df5 = pd.read_csv(file_path_diff5)

    ac_name += '-pred'
    specific_columns_df1 = df1[['Idx', 'Activity',ramdon_name]].rename(columns={ramdon_name: 'ram_pred'})
    specific_columns_df2 = df2[['Idx', ac_name]].rename(columns={ac_name: 'ac_pred_df'})
    specific_columns_df3 = df3[['Idx', ac_name]].rename(columns={ac_name: 'ac_pred_df'})
    specific_columns_df4 = df4[['Idx', ac_name]].rename(columns={ac_name: 'ac_pred_df'})
    specific_columns_df5 = df5[['Idx', ac_name]].rename(columns={ac_name: 'ac_pred_df'})

    combined_df2 = pd.merge(specific_columns_df1, specific_columns_df2, on='Idx',how='inner')
    combined_df3 = pd.merge(specific_columns_df1, specific_columns_df3, on='Idx', how='inner')
    combined_df4 = pd.merge(specific_columns_df1, specific_columns_df4, on='Idx', how='inner')
    combined_df5 = pd.merge(specific_columns_df1, specific_columns_df5, on='Idx', how='inner')

    return combined_df2, combined_df3, combined_df4, combined_df5

def cal_recall(y_pred, y_true, top):
    a_sort_idx = y_pred.argsort()
    b_sort_idx = y_true.argsort()
    recall = len(set(b_sort_idx[-top:].tolist()).intersection(a_sort_idx[-top:].tolist()))
    return recall

def cal_auc(predicted_values, true_values):
    recall = cal_recall(predicted_values, true_values, 50)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    spearman_corr, _ = spearmanr(true_values, predicted_values)
    pearson_corr, _ = pearsonr(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return recall, rmse, spearman_corr, pearson_corr, mae, r2

def cal_diff(df,model_name):

    recall_diff = []
    RMSE_diff = []
    spearman_diff = []
    pearson_diff = []
    mae_diff = []
    r2_diff = []

    Recall_ram = []
    RMSE_ram = []
    Spearman_ram = []
    Pearson_ram = []
    Mae_ram = []
    R2_ram = []

    Recall_ac = []
    RMSE_ac = []
    Spearman_ac = []
    Pearson_ac = []
    Mae_ac = []
    R2_ac = []

    for i in range(4):
        recall_ram, rmse_ram, spearman_corr_ram, pearson_corr_ram, mae_ram, r2_ram = cal_auc(df[i]['Activity'],
                                                                                             df[i]['ram_pred'])
        recall_ac,rmse_ac,spearman_corr_ac,pearson_corr_ac,mae_ac,r2_ac = cal_auc(df[i]['Activity'],df[i][f'ac_pred_df'])

        Recall_ram.append(recall_ram)
        RMSE_ram.append(rmse_ram)
        Spearman_ram.append(spearman_corr_ram)
        Pearson_ram.append(pearson_corr_ram)
        Mae_ram.append(mae_ram)
        R2_ram.append(r2_ram)

        Recall_ac.append(recall_ac)
        RMSE_ac.append(rmse_ac)
        Spearman_ac.append(spearman_corr_ac)
        Pearson_ac.append(pearson_corr_ac)
        Mae_ac.append(mae_ac)
        R2_ac.append(r2_ac)

        recall_diff.append(recall_ram-recall_ac)
        RMSE_diff.append(rmse_ram-rmse_ac)
        spearman_diff.append(spearman_corr_ram-spearman_corr_ac)
        pearson_diff.append(pearson_corr_ram-pearson_corr_ac)
        mae_diff.append(mae_ram-mae_ac)
        r2_diff.append(r2_ram-r2_ac)

    recall_ram_res[model_name] = (Recall_ram)
    RMSE_ram_res[model_name] = (RMSE_ram)
    spearman_ram_res[model_name] = (Spearman_ram)
    pearson_ram_res[model_name] = (Pearson_ram)
    mae_ram_res[model_name] = (Mae_ram)
    r2_ram_res[model_name] = (R2_ram)

    recall_ac_res[model_name] = (Recall_ac)
    RMSE_ac_res[model_name] = (RMSE_ac)
    spearman_ac_res[model_name] = (Spearman_ac)
    pearson_ac_res[model_name] = (Pearson_ac)
    mae_ac_res[model_name] = (Mae_ac)
    r2_ac_res[model_name] = (R2_ac)

    recall_df_res[model_name] = (recall_diff)
    RMSE_df_res[model_name] = (RMSE_diff)
    spearman_df_res[model_name] = (spearman_diff)
    pearson_df_res[model_name] = (pearson_diff)
    mae_df_res[model_name] = (mae_diff)
    r2_df_res[model_name] = (r2_diff)

model_names = ['esm2_t6', 'esm2_t12','esm2_t33','progen2-base','progen2-medium']

recall_df_res = {model: [] for model in model_names}
RMSE_df_res = {model: [] for model in model_names}
spearman_df_res = {model: [] for model in model_names}
pearson_df_res = {model: [] for model in model_names}
mae_df_res = {model: [] for model in model_names}
r2_df_res = {model: [] for model in model_names}

recall_ram_res = {model: [] for model in model_names}
RMSE_ram_res = {model: [] for model in model_names}
spearman_ram_res = {model: [] for model in model_names}
pearson_ram_res = {model: [] for model in model_names}
mae_ram_res = {model: [] for model in model_names}
r2_ram_res = {model: [] for model in model_names}

recall_ac_res = {model: [] for model in model_names}
RMSE_ac_res = {model: [] for model in model_names}
spearman_ac_res = {model: [] for model in model_names}
pearson_ac_res = {model: [] for model in model_names}
mae_ac_res = {model: [] for model in model_names}
r2_ac_res = {model: [] for model in model_names}


for model_name in model_names:
    df_act = combination(model_name)
    cal_diff(df_act,model_name)

writer = pd.ExcelWriter('valid-DL_metrics_df_results.xlsx', engine='openpyxl')
for metric_name, metric_dict in [
    ('recall', recall_df_res),
    ('RMSE', RMSE_df_res),
    ('spearman', spearman_df_res),
    ('pearson', pearson_df_res),
    ('mae', mae_df_res),
    ('r2', r2_df_res)]:
    df = pd.DataFrame(metric_dict).T  # 转置以模型名为索引
    df.index.name = 'Model'
    df.to_excel(writer, sheet_name=metric_name)  # 写入不同工作表
writer._save()  # 保存文件

writer = pd.ExcelWriter('valid-DL_metrics_ram_results.xlsx', engine='openpyxl')
for metric_name, metric_dict in [
    ('recall', recall_ram_res),
    ('RMSE', RMSE_ram_res),
    ('spearman', spearman_ram_res),
    ('pearson', pearson_ram_res),
    ('mae', mae_ram_res),
    ('r2', r2_ram_res)]:
    df = pd.DataFrame(metric_dict).T  # 转置以模型名为索引
    df.index.name = 'Model'
    df.to_excel(writer, sheet_name=metric_name)  # 写入不同工作表
writer._save()  # 保存文件

writer = pd.ExcelWriter('valid-DL_metrics_ac_results.xlsx', engine='openpyxl')
for metric_name, metric_dict in [
    ('recall', recall_ac_res),
    ('RMSE', RMSE_ac_res),
    ('spearman', spearman_ac_res),
    ('pearson', pearson_ac_res),
    ('mae', mae_ac_res),
    ('r2', r2_ac_res)]:
    df = pd.DataFrame(metric_dict).T  # 转置以模型名为索引
    df.index.name = 'Model'
    df.to_excel(writer, sheet_name=metric_name)  # 写入不同工作表
writer._save()  # 保存文件
print('done')

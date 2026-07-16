# maintained by kewei li
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import KMeans,DBSCAN
import os
import ipdb


def random_split_data(data_path, fold, output_dir):
    """
    将输入的数据框随机分成k组

    参数:
    df: Pandas 数据框
    k: 将数据框分成的组数

    返回:
    带有新列 'group' 的数据框，表示每行的组别
    """
    df = pd.read_csv(data_path)
    # 创建 KFold 实例
    kf = KFold(n_splits=fold, shuffle=True, random_state=42)
    os.makedirs(output_dir, exist_ok=True)
    # 初始化一个空的列来存储组别信息
    df['group'] = None

    # 为每个分割的组分配组别编号
    for group_number, (train_index, test_index) in enumerate(kf.split(df)):
        df.iloc[test_index, df.columns.get_loc('group')] = group_number

        train, test = df.iloc[train_index], df.iloc[test_index]
        # ipdb.set_trace()
        # Save to CSV
        train.to_csv(f'{output_dir}/train_fold_{group_number}.csv', index=False)
        test.to_csv(f'{output_dir}/test_fold_{group_number}.csv', index=False)
    

def stratified_split_data(data_path, fold, output_dir):
    """
    将输入的数据框随机分成k组

    参数:
    df: Pandas 数据框
    k: 将数据框分成的组数

    返回:
    带有新列 'group' 的数据框，表示每行的组别
    """
    df = pd.read_csv(data_path)

    z = df['Activity'].values
    kmeans = KMeans(n_clusters=fold, random_state=42, n_init=fold).fit(z.reshape(-1,1))
    
    clusters = kmeans.labels_
    # 创建 KFold 实例
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    os.makedirs(output_dir, exist_ok=True)
    # 初始化一个空的列来存储组别信息
    df['group'] = None

    # 为每个分割的组分配组别编号
    for group_number, (train_index, test_index) in enumerate(kf.split(df,clusters)):
        df.iloc[test_index, df.columns.get_loc('group')] = group_number

        train, test = df.iloc[train_index], df.iloc[test_index]

        # Save to CSV
        train.to_csv(f'{output_dir}/train_fold_{group_number}.csv', index=False)
        test.to_csv(f'{output_dir}/test_fold_{group_number}.csv', index=False)
    

def cluster_split_data(data_path, fold, output_dir):
    """
    将输入的数据框随机分成k组

    参数:
    df: Pandas 数据框
    k: 将数据框分成的组数

    返回:
    带有新列 'group' 的数据框，表示每行的组别
    """
    df = pd.read_csv(data_path)

    clusters = np.unique(df['levenshtein distance'].values)-1
    # 初始化一个空的列来存储组别信息
    df['group'] = df['levenshtein distance']-1

    # 为每个分割的组分配组别编号
    for c in clusters:
        train_index = df['group'] == c
        test_index = df['group'] != c
        train, test = df[train_index], df[test_index]

        # Save to CSV
        train.to_csv(f'{output_dir}/train_fold_{c}.csv', index=False)
        test.to_csv(f'{output_dir}/test_fold_{c}.csv', index=False)
    

def fixed_cluster_split_data(data_path, fold, output_dir):
    
    df = pd.read_csv(data_path)
    clusters = np.unique(df['fold'].values)
    os.makedirs(output_dir,exist_ok=True)
    # 为每个分割的组分配组别编号
    for c in clusters:
        train_index = df['fold'] == c
        test_index = df['fold'] != c
        train, test = df[train_index], df[test_index]

        # Save to CSV
        train.to_csv(f'{output_dir}/train_fold_{int(c)}.csv', index=False)
        test.to_csv(f'{output_dir}/test_fold_{int(c)}.csv', index=False)
    



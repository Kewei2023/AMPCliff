# maintained by kewei li
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Sequence, Tuple
def reorder_by_mic(csv_path: str, out_path: str = None):
    # 读取并仅保留需要的列；自动识别分隔符（逗号/制表符等）
    df = pd.read_csv(csv_path, sep=None, engine="python", usecols=["seq1","seq2","mic1","mic2"]).copy()

    # 数值化，避免字符串/科学计数法导致比较异常
    df["mic1"] = pd.to_numeric(df["mic1"], errors="coerce")
    df["mic2"] = pd.to_numeric(df["mic2"], errors="coerce")

    # 需要交换的位置：mic1 < mic2
    mask = df["mic1"] < df["mic2"]

    # 交换 mic
    df.loc[mask, ["mic1", "mic2"]] = df.loc[mask, ["mic2", "mic1"]].to_numpy()
    # 同步交换对应的序列
    df.loc[mask, ["seq1", "seq2"]] = df.loc[mask, ["seq2", "seq1"]].to_numpy()

    if out_path:
        df.to_csv(out_path, index=False)
    return df

def pairwise_accuracy_from_seq_preds(
    pairs_csv: str,                 # 上一步输出：含 seq1,seq2,mic1,mic2，且已保证 mic1>mic2
    per_seq_csv: str,               # 本步输入：单序列预测结果表
    out_pairs_with_preds: str = "",  # 可选：保存“对格式+各模型预测”的表
    out_acc: str = "",          # 可选：保存各模型准确率字典到 CSV
    diff: int = 5,                     # MIC 差值阈值（倍数），默认 5 倍
    pred_columns: Optional[Sequence[str]] = None,  # 若给定，仅计算这些预测列（须存在于 per_seq_csv）
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # 1) 读取并只保留需要列，保持 pair 顺序
    pairs = pd.read_csv(pairs_csv, sep=None, engine="python",
                        usecols=["seq1", "seq2", "mic1", "mic2"]).copy()

    # 2) 读取单序列预测；默认挑出固定多模型列；pred_columns 非空时仅使用指定列
    seq_df = pd.read_csv(per_seq_csv, sep=None, engine="python")
    if "sequence" in seq_df.columns and "Sequence" not in seq_df.columns:
        seq_df = seq_df.rename(columns={"sequence": "Sequence"})

    if pred_columns is not None:
        pred_cols: List[str] = list(pred_columns)
        if not pred_cols:
            raise ValueError("pred_columns must be non-empty when provided")
        missing = [c for c in pred_cols if c not in seq_df.columns]
        if missing:
            raise ValueError(f"pred_columns not found in per_seq_csv: {missing}")
    else:
        pred_cols = [c for c in seq_df.columns
                     if c in ["esm2_t6-pred", "esm2-t12-pred",
                              "SVM-pred", "LR-pred", "L1-pred", "L2-pred",
                              "ElasticNet-pred", "RF-pred", "GB-pred", "XGBoost-pred", "GP-pred"
                              ]]

    # 若同一 Sequence 出现多次（少见），取数值列均值聚合
    seq_map = (seq_df[["Sequence"] + pred_cols]
               .groupby("Sequence", as_index=False).mean(numeric_only=True))

    # 3) 两次左连接：把各模型在 seq1/seq2 上的预测并到 pair 表（加 _1/_2 后缀区分）
    s1 = seq_map.rename(columns={"Sequence": "seq1", **{c: f"{c}_1" for c in pred_cols}})
    s2 = seq_map.rename(columns={"Sequence": "seq2", **{c: f"{c}_2" for c in pred_cols}})
    pairs_pred = pairs.merge(s1, on="seq1", how="left").merge(s2, on="seq2", how="left")

    # 4) 逐模型计算准确率：严格比较 > ，缺失值对被跳过
    acc = {}
    for c in pred_cols:
        a = pairs_pred[f"{c}_1"]; b = pairs_pred[f"{c}_2"]
        m = a.notna() & b.notna()
        n = int(m.sum())
        acc[c] = float(((a[m]-b[m])>np.log10(diff)).sum()) / n if n > 0 else float("nan")
    # 5) 保存准确率字典到 CSV
    if out_acc:
        os.makedirs(os.path.dirname(out_acc),exist_ok=True)
        pd.DataFrame.from_dict(acc, orient='index', columns=['accuracy']).sort_values(by='accuracy',ascending=False).to_csv(out_acc)
    # 6) 可选：保存“对格式+预测值”的表
    if out_pairs_with_preds:
        os.makedirs(os.path.dirname(out_pairs_with_preds),exist_ok=True)
        pairs_pred.to_csv(out_pairs_with_preds, index=False)

    return pairs_pred, acc


if __name__=='__main__':
    # -------------- 使用示例 --------------
    '''
    e_coli 7-25 AC pairs, BLOSUM62 average, 5-fold CV
    '''
    print("****E COLI****")
    # print("====diff 5 BAGUA V1====")
    reorder_by_mic(csv_path="/data/run01/scv6872/kwli/AMPCliff/data/blast/blosum62 average/grampa_e_coli_7_25_acpairs_blosum62 average_5-fold.csv",
                     out_path="/data/home/scv6872/run/kwli/AMPCliff/data/blast/reordered_e_coli_pairs_5-fold.csv")
    pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
        pairs_csv="/data/run01/scv6872/kwli/AMPCliff/data/blast/reordered_e_coli_pairs_5-fold.csv",
        per_seq_csv="/data/run01/scv6872/kwli/AMPCliff/outputs/2025-11-13/20-13-55/test_pred_results_org_and_taichci-diff5.csv",
        out_pairs_with_preds="/data/home/scv6872/run/kwli/AMPCliff/learned_bagua/v1/diff5-pairs_with_all_model_preds_e_coli.csv",
        out_acc="./taichinet/distill-vc/acc_dict_diff5_e_coli.csv",
        diff=5
    )
    print(pd.Series(acc_dict).sort_values(ascending=False))
    print("****S AUREUS****")
    # print("====diff 5 BAGUA BASELINE====")
    pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
        pairs_csv="/data/home/scv6872/run/kwli/AMPCliff/data/reordered_s_aureus_pairs_5-fold.csv",
        per_seq_csv="/data/run01/scv6872/kwli/AMPCliff/outputs/2025-11-13/20-07-20/test_pred_results_org_and_taichci-diff5.csv",
        out_pairs_with_preds="./learned_bagua/v1/diff5-pairs_with_all_model_preds_s_aureus.csv",
        out_acc="./taichinet/distill-vc/acc_dict_diff5_s_aureus.csv",
        diff=5
    )
    print(pd.Series(acc_dict).sort_values(ascending=False))
    
    '''
    print("****E COLI****")
    print("====diff 5 BAGUA V1====")
    reorder_by_mic(csv_path="/data/home/scv6872/run/kwli/AMPCliff/data/blast/blosum62 average/grampa_e_coli_7_25_acpairs_blosum62 average_5-fold.csv",
                     out_path="/data/home/scv6872/run/kwli/AMPCliff/data/blast/reordered_e_coli_pairs_5-fold.csv")
    pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
        pairs_csv="/data/home/scv6872/run/kwli/AMPCliff/data/blast/reordered_e_coli_pairs_5-fold.csv",
        per_seq_csv="/data/home/scv6872/run/kwli/AMPCliff/outputs/2025-10-19/17-27-49/test_pred_results_org_and_taichci-diff5.csv",
        out_pairs_with_preds="/data/home/scv6872/run/kwli/AMPCliff/learned_bagua/v1/diff5-pairs_with_all_model_preds_e_coli.csv",
        out_acc="/data/home/scv6872/run/kwli/AMPCliff/learned_bagua/v1/acc_dict_diff5_e_coli.csv",
        diff=5
    )
    print(pd.Series(acc_dict).sort_values(ascending=False))
    print("====diff 5 BAGUA BASELINE====")
    pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
        pairs_csv="/data/home/scv6872/run/kwli/AMPCliff/data/blast/reordered_e_coli_pairs_5-fold.csv",
        per_seq_csv="/data/home/scv6872/run/kwli/AMPCliff/outputs/2025-10-20/08-07-32/test_pred_results_org_and_taichci-diff5.csv",
        out_pairs_with_preds="/data/home/scv6872/run/kwli/AMPCliff/learned_bagua/v1/diff5-pairs_with_all_model_preds_e_coli_baseline.csv",
        out_acc="/data/home/scv6872/run/kwli/AMPCliff/learned_bagua/v1/acc_dict_diff5_e_coli_baseline.csv",
        diff=5
    )
    print(pd.Series(acc_dict).sort_values(ascending=False))
    '''
    # print("====diff 5 BAGUA V1-1====")
    # # reorder_by_mic(csv_path="grampa_e_coli_7_25_acpairs_blosum62 average_5-fold.csv", out_path="e_coli_reordered_pairs_5-fold.csv")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="e_coli_reordered_pairs_5-fold.csv",
    #     per_seq_csv="./e_coli/bagua_v1-1/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./e_coli/bagua_v1-1/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./e_coli/bagua_v1-1/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 BAGUA V1-2====")
    # # reorder_by_mic(csv_path="grampa_e_coli_7_25_acpairs_blosum62 average_5-fold.csv", out_path="e_coli_reordered_pairs_5-fold.csv")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="e_coli_reordered_pairs_5-fold.csv",
    #     per_seq_csv="./e_coli/bagua_v1-2/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./e_coli/bagua_v1-2/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./e_coli/bagua_v1-2/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 BAGUA V1====")
    # # reorder_by_mic(csv_path="grampa_e_coli_7_25_acpairs_blosum62 average_5-fold.csv", out_path="e_coli_reordered_pairs_5-fold.csv")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="e_coli_reordered_pairs_5-fold.csv",
    #     per_seq_csv="./e_coli/bagua_v1/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./e_coli/bagua_v1/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./e_coli/bagua_v1/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))
    '''
    s.aureus 7-25 AC pairs, BLOSUM62 average, 5-fold CV
    '''
    print("****S AUREUS****")
    # print("====diff 5====")
    # reordered_pairs_df = reorder_by_mic(
    #     csv_path="/data/home/scv6872/run/kwli/AMPCliff/data/blast/blosum62 average/grampa_s_aureus_7_25_acpairs_blosum62 average_5-fold.csv", 
    #     out_path="/data/home/scv6872/run/kwli/AMPCliff/data/reordered_s_aureus_pairs_5-fold.csv"
    #     )
    '''
    pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
        pairs_csv="/data/home/scv6872/run/kwli/AMPCliff/data/reordered_s_aureus_pairs_5-fold.csv",
        per_seq_csv="/data/home/scv6872/run/kwli/AMPCliff/outputs/2025-10-19/10-51-55/test_pred_results_org_and_taichci-diff5.csv",
        out_pairs_with_preds="./learned_bagua/v1/diff5-pairs_with_all_model_preds_s_aureus.csv",
        out_acc="./learned_bagua/v1/acc_dict_diff5_s_aureus.csv",
        diff=5
    )
    print(pd.Series(acc_dict).sort_values(ascending=False))

    print("====diff 5 BASELINE====")
    pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
        pairs_csv="/data/home/scv6872/run/kwli/AMPCliff/data/reordered_s_aureus_pairs_5-fold.csv",
        per_seq_csv="/data/home/scv6872/run/kwli/AMPCliff/outputs/2025-10-20/07-59-12/test_pred_results_org_and_taichci-diff5.csv",
        out_pairs_with_preds="./learned_bagua/v1/diff5-pairs_with_all_model_preds_s_aureus_baseline.csv",
        out_acc="./learned_bagua/v1/acc_dict_diff5_s_aureus_baseline.csv",
        diff=5
    )
    print(pd.Series(acc_dict).sort_values(ascending=False))
    '''
    # print("====diff 5 FOURIER====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./fourier/test_pred_results_org_and_fourier-diff5.csv",
    #     out_pairs_with_preds="./fourier/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./fourier/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 BAGUA====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./bagua/test_pred_results_org_and_bagua-diff5.csv",
    #     out_pairs_with_preds="./bagua/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./bagua/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 Learned BAGUA time====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./learned_bagua_time/v1/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./learned_bagua_time/v1/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./learned_bagua_time/v1/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 Learned BAGUA PROJ TO REAL v1====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="/data/home/scv6872/run/kwli/AMPCliff/outputs/2025-10-18/20-08-07/esm2_t6-blosum62 average-diff5-trd0.9-test_result.csv",
    #     out_pairs_with_preds="./learned_bagua/v1/diff5-pairs_with_all_model_preds_s_aureus.csv",
    #     out_acc="./learned_bagua/v1/acc_dict_diff5_s_aureus.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 Learned BAGUA REAL V1-1====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./learned_bagua/v1-1-inter-t-intra-f/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./learned_bagua/v1-1-inter-t-intra-f/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./learned_bagua/v1-1-inter-t-intra-f/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 Learned BAGUA REAL V1-2====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./learned_bagua/v1-2/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./learned_bagua/v1-2/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./learned_bagua/v1-2/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 Learned BAGUA REAL V1-1 INTER TIME INTRA TIME====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./learned_bagua/v1-1-inter-t-intra-t/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./learned_bagua/v1-1-inter-t-intra-t/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./learned_bagua/v1-1-inter-t-intra-t/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 Learned BAGUA PROJ TO IMAGE V1====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./learned_bagua_image/v1_proj/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./learned_bagua_image/v1_proj/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./learned_bagua_image/v1_proj/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 Learned BAGUA REAL V1====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./learned_bagua/v1/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./learned_bagua/v1/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./learned_bagua/v1/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 Learned BAGUA REAL V3====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./learned_bagua/v3/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./learned_bagua/v3/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./learned_bagua/v3/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 5 Learned BAGUA REAL V4====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./learned_bagua/v4/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./learned_bagua/v4/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./learned_bagua/v4/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))
    
    # print("====diff 5 Learned BAGUA IMAGE V1====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_5-fold.csv",
    #     per_seq_csv="./learned_bagua_image/v1/test_pred_results_org_and_taichci-diff5.csv",
    #     out_pairs_with_preds="./learned_bagua_image/v1/diff5-pairs_with_all_model_preds.csv",
    #     out_acc="./learned_bagua_image/v1/acc_dict_diff5.csv",
    #     diff=5
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))
    # print("====diff 4====")
    # reordered_pairs_df = reorder_by_mic(csv_path="grampa_s_aureus_7_25_acpairs_blosum62 average_4-fold.csv", out_path="reordered_pairs_4-fold.csv")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_4-fold.csv",
    #     per_seq_csv="test_pred_results_org_and_taichci-diff4.csv",
    #     out_pairs_with_preds="diff4-pairs_with_all_model_preds.csv",
    #     out_acc="acc_dict_diff4.csv",
    #     diff=4
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))
    
    # print("====diff 4 BASELINE====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_4-fold.csv",
    #     per_seq_csv="./baseline/test_pred_results_org_and_taichci-diff4.csv",
    #     out_pairs_with_preds="./baseline/diff4-pairs_with_all_model_preds.csv",
    #     out_acc="./baseline/acc_dict_diff4.csv",
    #     diff=4
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 3====")
    # reordered_pairs_df = reorder_by_mic(csv_path="grampa_s_aureus_7_25_acpairs_blosum62 average_3-fold.csv", out_path="reordered_pairs_3-fold.csv")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_3-fold.csv",
    #     per_seq_csv="test_pred_results_org_and_taichci-diff3.csv",
    #     out_pairs_with_preds="diff3-pairs_with_all_model_preds.csv",
    #     out_acc="acc_dict_diff3.csv",
    #     diff=3
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

    # print("====diff 3 BASELINE====")
    # pairs_pred_df, acc_dict = pairwise_accuracy_from_seq_preds(
    #     pairs_csv="reordered_pairs_3-fold.csv",
    #     per_seq_csv="./baseline/test_pred_results_org_and_taichci-diff3.csv",
    #     out_pairs_with_preds="./baseline/diff3-pairs_with_all_model_preds.csv",
    #     out_acc="./baseline/acc_dict_diff3.csv",
    #     diff=3
    # )
    # print(pd.Series(acc_dict).sort_values(ascending=False))

探究过拟合问题

v1: 层间特征FFT之后做最大池化降维，层内特征FFT之后最大池化降维
v2: 层间特征FFT之后沿hidden_dim方向concat, *层内特征FFT之后取前K个幅度 concat 
v3: 层间特征FFT之后沿hidden_dim方向concat，层内特征FFT之后最大池化降维
v4: 层间特征FFT之后沿hidden_dim方向concat，层内特征FFT之后加线性层降维: /data/home/scv6872/run/kwli/AMPCliff/outputs/2025-08-27/00-41-30/esm2_t6/blosum62 average/diff5-trd0.9/model_step_43_spearman_0.726
v5:                                      层内特征FFT之后加线性层降维: /data/home/scv6872/run/kwli/AMPCliff/outputs/2025-08-27/16-58-04
v6: $层间特征FFT之后沿hidden_dim方向concat: /data/home/scv6872/run/kwli/AMPCliff/outputs/2025-08-27/17-35-47
v6: $层间特征FFT之后沿hidden_dim方向concat,减少旋转角度8->1: 
v7: 最后一层特征FFT之后加线性层降维
v1-1: 层间特征FFT之后做最大池化降维，然后转换到时域，层内特征FFT之后最大池化降维，然后转换到时域
*被证实效果不行

不同靶标细菌

e_coli real: /data/home/scv6872/run/kwli/AMPCliff/outputs/2025-08-28/14-50-20
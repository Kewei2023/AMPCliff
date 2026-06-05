# Local spectral pooling 消融：命令与结论占位

**实现侧自检**：四种 pooling（`mean` / `attn` / `multi_head_spectral` / `local_spectral_anchor`）在 `ClassificationHead2` 上前向已通过；完整训练对比需在目标 GPU/数据路径上跑 `downstream_train.py` 后填写下表。

## 四组对比（AMP 回归，Hydra 覆盖）

在仓库根目录、且 `PYTHONPATH` 包含 AMPCliff 的父目录（例如 `export PYTHONPATH=/mnt/d`）时，可用同一随机种子与数据划分切换 pooling：

```bash
# 1) mean baseline
python downstream_train.py model.regression.pooling=mean

# 2) attention pooling
python downstream_train.py model.regression.pooling=attn

# 3) 旧版多头频谱（全局 rFFT + 插值）
python downstream_train.py model.regression.pooling=multi_head_spectral model.regression.use_fft=true

# 4) 新版局部 STFT + overlap-add
python downstream_train.py model.regression.pooling=local_spectral_anchor \
  model.regression.analysis_dim=8 model.regression.stft_n_fft=8 \
  model.regression.stft_win_length=8 model.regression.stft_hop_length=4 \
  model.regression.stft_center=false model.regression.use_phase=false
```

建议固定：`train.random_seed`、`data.regression.*`、`model.regression.apply` 等与论文表格一致；输出目录由 `hydra.run.dir` 自动带上 `pooling` 名称。

## 前向冒烟（不训练）

验证四种 pooling 在 `ClassificationHead2` 上可跑通前向（实现自检）：

```bash
cd /path/to/AMPCliff
PYTHONPATH=/path/to/parent_of_AMPCliff python - <<'PY'
from types import SimpleNamespace
import torch
from AMPCliff.factory.regression import ClassificationHead2

def cfg(pooling):
    return SimpleNamespace(
        hidden_size=64,
        hidden_dropout_prob=0.0,
        num_labels=1,
        pooling=pooling,
        num_anchor=8,
        num_heads=4,
        use_fft=True,
        aggregation="soft",
        use_projection=True,
        gated=True,
        dropout=0.0,
        num_freq_components=16,
        analysis_dim=8,
        stft_n_fft=8,
        stft_win_length=8,
        stft_hop_length=4,
        stft_center=False,
        use_phase=False,
    )

for p in ("mean", "attn", "multi_head_spectral", "local_spectral_anchor"):
    head = ClassificationHead2(cfg(p))
    x, m = torch.randn(2, 17, 64), torch.ones(2, 17)
    y = head(x, m)
    assert y.shape == (2, 1) and torch.isfinite(y).all(), p
print("forward smoke: ok")
PY
```

## 简短结论（训练后填写）

| Pooling | valid spearman | test spearman | 备注 |
|---------|----------------|---------------|------|
| mean | | | |
| attn | | | |
| multi_head_spectral | | | 旧版全局频谱插值 |
| local_spectral_anchor | | | STFT 帧级 + overlap-add |

完整指标可继续用项目内实验统计脚本/技能产出 CSV。

# Local spectral pooling 改造 — 旧版基线（baseline-freeze）

用于与 `local_spectral_anchor` 对比时复现同一设置。

## Baseline：`multi_head_spectral`（全局 rFFT + 线性插值回序列）

- **实现**：[`MultiHeadSpectralAnchorPooling`](factory/pooling/spectral_anchor_v2.py)（`use_fft=true` 时 `rfft` → `spectral_transform` → anchor softmax → `F.interpolate` 对齐 `seq_len`）。
- **典型 Hydra 配置**（见 [`configs/downstream.yaml`](../configs/downstream.yaml) 同类字段）：
  - `model.regression.pooling: multi_head_spectral`
  - `model.regression.use_fft: true`
  - `model.regression.num_anchor: 8`（经 registry 映射为 `num_anchor_per_head = num_anchor // num_heads`，若 `num_anchor < num_heads` 则为 `2`）
  - `model.regression.num_heads: 4`
  - `model.regression.gated: true`
  - `model.regression.dropout: 0.0`
- **结果路径**：由 Hydra `hydra.run.dir` 生成，目录名含 `${model.regression.pooling}`；具体 `outputs/...` 路径以每次运行为准。

## 新版对照：`local_spectral_anchor`

- **实现**：`MultiHeadLocalSpectralAnchorPooling`（STFT 帧级 anchor → overlap-add 回 token 权重，无频率轴线性插值）。
- **配置**：`model.regression.pooling: local_spectral_anchor`，并设置 `analysis_dim`、`stft_n_fft`、`stft_win_length`、`stft_hop_length`、`use_phase`、`stft_center`（见 `downstream.yaml` 注释块）。

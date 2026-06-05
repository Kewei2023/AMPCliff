# Spectral Anchor Pooling 分析与优化方案

## 1. 实验数据总结

### ESM2-T6 测试集 Spearman 对比

| Dataset | mean | max | attn | spectral_anchor (best) | 最佳配置 |
|---------|------|-----|------|------------------------|----------|
| e_coli | 0.5325 | 0.5359 | 0.5229 | 0.5346 | k2_fftTrue |
| s_aureus | 0.3937 | 0.4142 | 0.4378 | **0.4503** | k8_fftFalse |

**关键发现:**
- 对于 s_aureus, spectral_anchor 已经超越所有baseline
- 对于 e_coli, spectral_anchor 与 baseline 持平
- use_fft=False 的配置在某些情况下表现更好 (s_aureus: k8_fftFalse)

## 2. 当前实现分析

### 2.1 核心代码流程
```python
# 1. FFT变换 (可选)
spectral = torch.fft.rfft(x, dim=1).real  # 只取实部!

# 2. 计算与anchors的距离
dist = torch.cdist(spectral, anchors)
alpha = torch.softmax(-dist, dim=-1)

# 3. 取最大anchor权重作为token saliency
weight = alpha.max(dim=-1).values

# 4. 插值回原始长度 (FFT会改变序列长度)
weight = F.interpolate(weight, size=seq_len, mode="linear")

# 5. 加权求和
pooled = (x * weight).sum(dim=1)
```

### 2.2 问题诊断

#### 问题1: FFT只取实部丢失信息
```python
spectral = torch.fft.rfft(x, dim=1).real  # 丢失虚部(相位信息)
```
- rfft 输出是复数，包含幅度和相位
- 只取实部会丢失约50%的频域信息
- 相位信息对于序列结构很重要

#### 问题2: 简单的距离度量
```python
dist = torch.cdist(spectral, anchors)  # 欧氏距离
```
- 频域中欧氏距离可能不是最优度量
- 没有考虑不同频率分量的重要性差异

#### 问题3: 信息聚合不足
```python
weight = alpha.max(dim=-1).values  # 只取最大值
```
- 丢弃了其他anchor的信息
- 类似于hard attention，可能不是最优

#### 问题4: 插值引入误差
```python
weight = F.interpolate(weight, size=seq_len, mode="linear")
```
- FFT将序列长度从T变为T//2+1
- 线性插值会引入平滑误差

#### 问题5: 与Attention Pooling的架构差距

**Attention Pooling (VectorAttentionHead):**
```python
S = W1(X)           # 线性变换
S1 = relu(W2(S))    # 非线性变换
e = W3(S1)          # 再次变换
if gated:
    e = e * sigmoid(U(X))  # 门控机制
A = softmax(e)      # 注意力权重
v = (A * S).sum()   # 加权聚合
```

**Spectral Anchor Pooling:**
```python
spectral = fft(x).real  # 简单FFT
dist = cdist(spectral, anchor)  # 距离计算
weight = softmax(-dist).max()   # 简单聚合
pooled = (x * weight).sum()     # 加权求和
```

**对比分析:**
| 特性 | Attention Pooling | Spectral Anchor |
|------|------------------|-----------------|
| 参数变换 | 3层MLP + 门控 | 无 |
| 非线性 | ReLU | 无 |
| 学习能力 | 高 | 仅anchor可学习 |
| 多头机制 | 支持 | 不支持 |

## 3. 优化方案

### 方案1: 增强型频谱表示 (Enhanced Spectral Representation)

```python
class EnhancedSpectralAnchorPooling(nn.Module):
    """使用完整的频谱信息"""

    def __init__(self, d_model, num_anchor=8, use_fft=True):
        super().__init__()
        self.num_anchor = num_anchor
        self.use_fft = use_fft

        # 频谱投影层
        self.spectral_proj = nn.Linear(d_model * 2, d_model)  # 处理幅度+相位

        # 可学习anchors
        self.anchor = nn.Parameter(torch.empty(num_anchor, d_model))
        nn.init.xavier_uniform_(self.anchor)

        # 权重变换网络
        self.weight_transform = nn.Sequential(
            nn.Linear(num_anchor, num_anchor * 2),
            nn.ReLU(),
            nn.Linear(num_anchor * 2, 1)
        )

    def _compute_spectral_features(self, x):
        if not self.use_fft:
            return x

        # 完整FFT
        fft_result = torch.fft.rfft(x, dim=1)
        magnitude = torch.abs(fft_result)  # 幅度
        phase = torch.angle(fft_result)    # 相位

        # 拼接幅度和相位
        spectral = torch.cat([magnitude, phase], dim=-1)
        return self.spectral_proj(spectral)
```

### 方案2: 多头频谱注意力 (Multi-Head Spectral Attention)

```python
class MultiHeadSpectralAnchorPooling(nn.Module):
    """借鉴Attention Pooling的多头设计"""

    def __init__(self, d_model, num_heads=4, num_anchor=8, use_fft=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_anchor = num_anchor

        # 每个头独立的anchor
        self.anchors = nn.Parameter(torch.empty(num_heads, num_anchor, d_model // num_heads))
        nn.init.xavier_uniform_(self.anchors)

        # 频谱变换
        self.spectral_transform = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # 门控机制
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape

        # 频谱变换
        if self.use_fft:
            spectral = torch.fft.rfft(x, dim=1)
            x_freq = torch.cat([spectral.real, spectral.imag], dim=-1)
            x_freq = self.spectral_transform(x_freq[:, :, :D])
        else:
            x_freq = x

        # 多头处理
        head_outputs = []
        for h in range(self.num_heads):
            x_h = x_freq.view(B, T, self.num_heads, -1)[:, :, h, :]
            anchors_h = self.anchors[h]

            # 计算注意力
            attn = torch.softmax(-torch.cdist(x_h, anchors_h), dim=-1)
            weight = attn.mean(dim=-1)  # 使用mean而不是max
            head_out = (x_h * weight.unsqueeze(-1)).sum(dim=1)
            head_outputs.append(head_out)

        # 合并多头
        pooled = torch.cat(head_outputs, dim=-1)

        # 门控输出
        gate = torch.sigmoid(self.gate(pooled))
        return self.output_proj(pooled * gate)
```

### 方案3: 频率加权池化 (Frequency-Weighted Pooling)

```python
class FrequencyWeightedPooling(nn.Module):
    """直接学习频率分量的重要性"""

    def __init__(self, d_model, num_freq_bins=None, use_fft=True):
        super().__init__()
        self.use_fft = use_fft
        self.d_model = d_model

        if use_fft:
            # 学习每个频率bin的重要性
            self.freq_importance = nn.Parameter(torch.ones(1, 1, 1))
            self.freq_transform = nn.Linear(d_model * 2, d_model)

        # 时域权重网络
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape

        if self.use_fft:
            # FFT并保留完整信息
            fft_result = torch.fft.rfft(x, dim=1)
            freq_features = torch.cat([fft_result.real, fft_result.imag], dim=-1)
            freq_features = self.freq_transform(freq_features[:, :, :D])

            # 学习频率权重
            freq_weights = torch.softmax(self.freq_importance, dim=-1)
            freq_pooled = (freq_features * freq_weights).mean(dim=1)

            # 逆变换回时域权重
            temporal_weights = self.temporal_attention(x)
        else:
            temporal_weights = self.temporal_attention(x)
            freq_pooled = x.mean(dim=1)

        # 合并时域和频域信息
        if attention_mask is not None:
            temporal_weights = temporal_weights.masked_fill(~attention_mask.bool(), -1e9)

        temporal_weights = torch.softmax(temporal_weights, dim=1)
        temporal_pooled = (x * temporal_weights).sum(dim=1)

        return temporal_pooled + freq_pooled
```

### 方案4: 渐进式优化 (推荐: 最小改动方案)

基于现有代码的最小改动优化:

```python
class SpectralAnchorPoolingV2(nn.Module):
    """渐进式优化版本 - 保持简单性的同时提升性能"""

    def __init__(self, d_model: int, num_anchor: int = 8, use_fft: bool = True):
        super().__init__()
        self.num_anchor = int(num_anchor)
        self.use_fft = bool(use_fft)

        # 优化1: 添加输入变换
        self.input_proj = nn.Linear(d_model, d_model)

        # 优化2: anchors初始化改进
        self.anchor = nn.Parameter(torch.empty(self.num_anchor, d_model))
        nn.init.orthogonal_(self.anchor)  # 正交初始化,增加多样性

        # 优化3: 添加温度参数
        self.temperature = nn.Parameter(torch.ones(1))

        # 优化4: 添加输出变换
        self.output_proj = nn.Linear(d_model, d_model)

    def _compute_assignment_weights(self, x: torch.Tensor) -> torch.Tensor:
        # 应用输入变换
        x = self.input_proj(x)

        if self.use_fft:
            # 优化5: 使用完整的频谱信息
            fft_result = torch.fft.rfft(x, dim=1)
            # 使用幅度谱(比只用实部更稳定)
            spectral = torch.abs(fft_result)
        else:
            spectral = x

        # 优化6: 缩放距离,增加数值稳定性
        anchors = self.anchor.unsqueeze(0).expand(spectral.size(0), -1, -1)
        dist = torch.cdist(spectral, anchors) / (spectral.size(-1) ** 0.5)

        # 优化7: 使用可学习的温度
        alpha = torch.softmax(-dist / self.temperature.abs(), dim=-1)

        # 优化8: 使用加权平均而非max
        weight = (alpha * torch.softmax(-dist, dim=-1)).sum(dim=-1)
        return weight

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        weight = self._compute_assignment_weights(x)

        # 处理FFT导致的长度变化
        if weight.size(1) != seq_len:
            weight = F.interpolate(
                weight.unsqueeze(1),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=weight.dtype)
            weight = weight * mask

        weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (x * weight.view(batch_size, seq_len, 1)).sum(dim=1)

        return self.output_proj(pooled)
```

## 4. 实验建议

### 4.1 消融实验设计

1. **FFT表示方式对比:**
   - 只取实部 (当前)
   - 取幅度谱 (方案1)
   - 取幅度+相位 (方案2)

2. **权重聚合方式对比:**
   - max (当前)
   - mean
   - learned combination (方案4)

3. **添加变换层的影响:**
   - 无变换 (当前)
   - 输入变换
   - 输出变换
   - 两者都有

### 4.2 推荐的实验顺序

1. **快速验证:** 先测试方案4 (渐进式优化),改动最小
2. **深度优化:** 如果方案4效果不明显,测试方案2 (多头频谱注意力)
3. **最终方案:** 结合最佳组件创建最终版本

## 5. 代码实现位置

建议创建新文件:
- `factory/pooling/spectral_anchor_v2.py` - 渐进式优化版本
- `factory/pooling/multi_head_spectral.py` - 多头版本

修改注册:
- `factory/pooling/registry.py` - 添加新的pooling类型

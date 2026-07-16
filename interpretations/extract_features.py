import math
import torch
import torch.nn.functional as F
from collections import Counter
from typing import List, Dict, Tuple
from itertools import chain

# ===== 1) 基础表：氨基酸与理化属性 =====
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
AA_IDX = {a:i for i,a in enumerate(AA)}

# Kyte–Doolittle hydropathy
KD = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,
      'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
# Grantham side-chain volume
VOL = {'A':31,'C':55,'D':54,'E':83,'F':132,'G':3,'H':96,'I':111,'K':119,'L':111,'M':105,
       'N':56,'P':32.5,'Q':85,'R':124,'S':32,'T':61,'V':84,'W':170,'Y':136}
# Polarity (Zimmerman)
POL = {'A':8.1,'C':5.5,'D':13.0,'E':12.3,'F':5.2,'G':9.0,'H':10.4,'I':5.2,'K':11.3,'L':4.9,
       'M':5.7,'N':11.6,'P':8.0,'Q':10.5,'R':10.5,'S':9.2,'T':8.6,'V':5.9,'W':5.4,'Y':6.2}
# Coarse charge at pH 7
CHG = {a:0.0 for a in AA}; CHG.update({'D':-1,'E':-1,'K':+1,'R':+1,'H':+0.1})

# 分子量（平均原子量, Da）
MW_RES = {'A':71.0788,'C':103.1388,'D':115.0886,'E':129.1155,'F':147.1766,'G':57.0519,'H':137.1411,
          'I':113.1594,'K':128.1741,'L':113.1594,'M':131.1926,'N':114.1038,'P':97.1167,'Q':128.1307,
          'R':156.1875,'S':87.0782,'T':101.1051,'V':99.1326,'W':186.2132,'Y':163.1760}
H2O = 18.01528  # 肽链末端加水

# Chou–Fasman 常用集合（简化版）
CF_HELIX = set("EALMQKRH")
CF_STRAND = set("VIFYWT")
CF_TURN   = set("GPSTDN")

# pKa（近似）
PKA_SIDE = {'D':3.9,'E':4.1,'H':6.0,'C':8.3,'Y':10.1,'K':10.5,'R':12.5}
PKA_NTERM, PKA_CTERM = 8.0, 3.1
PH = 7.0

# ===== 2) 工具函数 =====
def map_prop(seq: str, table: Dict[str, float]) -> torch.Tensor:
    vals = [table[a] for a in seq if a in AA_SET]
    return torch.tensor(vals, dtype=torch.float32)

def zscore(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    m, s = x.mean(), x.std(unbiased=False)
    return (x - m) / (s + 1e-8)

def approx_net_charge(seq: str, ph: float = PH) -> float:
    nterm = 1.0 / (1.0 + 10**(ph - PKA_NTERM))
    cterm = -1.0 / (1.0 + 10**(PKA_CTERM - ph))
    charge = nterm + cterm
    for a in seq:
        if a == 'D':
            charge += -1.0 / (1.0 + 10**(PKA_SIDE['D'] - ph))
        elif a == 'E':
            charge += -1.0 / (1.0 + 10**(PKA_SIDE['E'] - ph))
        elif a == 'C':
            charge += -1.0 / (1.0 + 10**(PKA_SIDE['C'] - ph))
        elif a == 'Y':
            charge += -1.0 / (1.0 + 10**(PKA_SIDE['Y'] - ph))
        elif a == 'H':
            charge += +1.0 / (1.0 + 10**(ph - PKA_SIDE['H']))
        elif a == 'K':
            charge += +1.0 / (1.0 + 10**(ph - PKA_SIDE['K']))
        elif a == 'R':
            charge += +1.0 / (1.0 + 10**(ph - PKA_SIDE['R']))
    return float(charge)

def estimate_pI(seq: str, pH_lo: float = 0.0, pH_hi: float = 14.0, tol: float = 1e-3) -> float:
    if not seq:
        return 0.0
    lo, hi = pH_lo, pH_hi
    q_lo = approx_net_charge(seq, lo)
    q_hi = approx_net_charge(seq, hi)
    if q_lo * q_hi > 0:
        return lo if abs(q_lo) < abs(q_hi) else hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        q_mid = approx_net_charge(seq, mid)
        if abs(q_mid) < tol:
            return mid
        if q_lo * q_mid <= 0:
            hi, q_hi = mid, q_mid
        else:
            lo, q_lo = mid, q_mid
    return 0.5 * (lo + hi)

def peptide_mass(seq: str) -> float:
    if not seq: return 0.0
    return float(sum(MW_RES[a] for a in seq) + H2O)

def aliphatic_index(seq: str) -> float:
    L = max(1, len(seq))
    f = Counter(seq)
    xA = f.get('A',0)/L
    xV = f.get('V',0)/L
    xIL = (f.get('I',0)+f.get('L',0))/L
    return 100.0*(xA + 2.9*xV + 3.9*xIL)

def dipeptide_freq(seq: str) -> torch.Tensor:
    L = len(seq)
    counts = torch.zeros((20,20), dtype=torch.float32)
    if L < 2:
        return counts.flatten()
    for i in range(L-1):
        a, b = seq[i], seq[i+1]
        if a in AA_SET and b in AA_SET:
            counts[AA_IDX[a], AA_IDX[b]] += 1.0
    return (counts / (L-1)).flatten()

def spectral_bands(x: torch.Tensor, freq_bands: List[Tuple[float,float]]) -> Tuple[torch.Tensor, float]:
    if x.numel() == 0:
        nb = len(freq_bands)
        return torch.zeros(nb, dtype=torch.float32), 0.0
    X = torch.fft.rfft(x)
    mag2 = (X.real**2 + X.imag**2)
    mag2 = mag2.clone(); mag2[0] = 0.0  # 去直流
    n_bins = mag2.numel()
    bands = []
    for lo, hi in freq_bands:
        i0 = max(1, int(math.floor(lo*(n_bins-1))))
        i1 = min(n_bins-1, int(math.ceil(hi*(n_bins-1))))
        if i1 < i0: i1 = i0
        bands.append(mag2[i0:i1+1].sum())
    bands = torch.stack(bands)
    if mag2.sum() > 0:
        idx = torch.arange(n_bins, dtype=mag2.dtype, device=mag2.device)
        centroid = float((idx * mag2).sum() / (mag2.sum() * (n_bins-1)))
    else:
        centroid = 0.0
    return bands, centroid

def short_autocorr(x: torch.Tensor, max_lag: int = 10) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(max_lag, dtype=torch.float32)
    x = x - x.mean()
    L = x.numel()
    ac = []
    for k in range(1, max_lag+1):
        n = L - k
        if n <= 0:
            ac.append(torch.tensor(0.0))
        else:
            ac.append((x[:n] * x[k:]).mean())
    return torch.stack(ac).to(torch.float32)

def window_std_stats(x: torch.Tensor, win: int = 8, stride: int = 1) -> Tuple[float,float]:
    L = x.numel()
    if L == 0:
        return 0.0, 0.0
    if L < win:
        s = float(x.std(unbiased=False))
        return s, s
    xs = x.unfold(0, size=win, step=stride)
    s = xs.std(dim=1, unbiased=False)
    return float(s.mean()), float(s.max())

def composition_entropy(cnt: Counter, L: int) -> float:
    if L <= 0: return 0.0
    ent = 0.0
    for a in AA:
        p = cnt.get(a, 0) / L
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return float(ent)

# ===== 3) 主过程：从 dataloader 生成两类特征（可直接替换） =====
def extract_features_from_dataloader(dataloader,
                                     use_dipeptide: bool = False,
                                     freq_bands: List[Tuple[float,float]] = [(0.05,0.20),(0.20,1.00)],
                                     autocorr_max_lag: int = 10,
                                     win: int = 8, stride: int = 1,
                                     device: str = "cpu"):
    """
    返回:
      global_feats:  (N, G)
      local_feats:   (N, Lc)
      global_names:  List[str]
      local_names:   List[str]
      ids:           List[any]
    """
    global_feat_list, local_feat_list, ids = [], [], []

    # ------ 全局特征名（扩充版） ------
    global_names = []
    global_names += ["length"]
    global_names += [f"aa_frac_{a}" for a in AA]  # 20

    # 组成分组
    global_names += [
        "frac_acidic(DE)", "frac_basic(KRH)", "frac_aromatic(FYW)", "frac_hydrophobic(AVLIMFWY)",
        "frac_polar(DENQKRHSTY)", "frac_nonpolar(ACGILMPVFYW)",
        "frac_tiny(ACGST)", "frac_small(ACDGNPSTV)",
        "frac_C", "has_disulfide_potential"
    ]

    # 理化均值与离散度
    global_names += ["kd_mean","kd_std","pol_mean","pol_std","vol_mean","vol_std"]

    # 电荷/酸碱
    global_names += ["net_charge_pH7","charge_density","pI"]

    # 质量/芳香/复杂度
    global_names += ["molecular_weight_Da","aromaticity_index(FYW)","composition_entropy"]

    # 传统指标
    global_names += ["aliphatic_index"]

    # 结构倾向（Chou–Fasman 简版）
    global_names += ["cf_frac_helixFormers","cf_frac_strandFormers","cf_frac_turnFormers"]

    # 二肽（可选 400 维）
    if use_dipeptide:
        global_names += [f"dipep_{a}{b}" for a in AA for b in AA]

    # ------ 局部/频域特征名 ------
    props = [("KD", KD), ("VOL", VOL), ("POL", POL), ("CHG", CHG)]
    band_names = [f"band_{int(lo*100)}_{int(hi*100)}" for (lo,hi) in freq_bands]
    local_names = []
    for p,_ in props:
        local_names += [f"{p}_spec_{bn}" for bn in band_names] + [f"{p}_spec_centroid"]
        local_names += [f"{p}_winstd_mean", f"{p}_winstd_max"]
        local_names += [f"{p}_ac_lag{lag}" for lag in range(1, autocorr_max_lag+1)]

    # ------ 数据遍历 ------
    for batch in dataloader:
        # 你的 dataloader 解包格式
        sequence, name2id, label = batch
        peptides = sequence["peptide"]
        # 聚合 ids（按你现有写法）
        ids.extend(list(chain.from_iterable(list(name2id.values()))))

        for seq in peptides:
            # 仅保留 20AA
            seq = "".join([a for a in seq if a in AA_SET])
            L = len(seq)

            # ======= 全局：扩充版 =======
            feat_g = []
            feat_g.append(float(L))

            # 20AA 频率
            if L > 0:
                cnt = Counter(seq)
                aa_fracs = [cnt.get(a,0)/L for a in AA]
            else:
                cnt = Counter()
                aa_fracs = [0.0]*20
            feat_g.extend(aa_fracs)

            # 组成分组
            frac_acid  = (seq.count('D') + seq.count('E')) / L if L>0 else 0.0
            frac_basic = (seq.count('K') + seq.count('R') + seq.count('H')) / L if L>0 else 0.0
            frac_aroma = sum(seq.count(a) for a in "FYW") / L if L>0 else 0.0
            frac_hydro = sum(seq.count(a) for a in "AVLIMFWY") / L if L>0 else 0.0
            frac_polar    = sum(seq.count(a) for a in "DENQKRHSTY") / L if L>0 else 0.0
            frac_nonpolar = sum(seq.count(a) for a in "ACGILMPVFYW") / L if L>0 else 0.0
            frac_tiny  = sum(seq.count(a) for a in "ACGST") / L if L>0 else 0.0
            frac_small = sum(seq.count(a) for a in "ACDGNPSTV") / L if L>0 else 0.0
            frac_cys   = seq.count('C') / L if L>0 else 0.0
            has_ss     = 1.0 if seq.count('C') >= 2 else 0.0
            feat_g += [frac_acid, frac_basic, frac_aroma, frac_hydro,
                       frac_polar, frac_nonpolar, frac_tiny, frac_small, frac_cys, has_ss]

            # KD/POL/VOL 均值与标准差
            if L > 0:
                vals_kd  = torch.tensor([KD[a]  for a in seq], dtype=torch.float32)
                vals_pol = torch.tensor([POL[a] for a in seq], dtype=torch.float32)
                vals_vol = torch.tensor([VOL[a] for a in seq], dtype=torch.float32)
                gravy_mean = float(vals_kd.mean())
                gravy_std  = float(vals_kd.std(unbiased=False)) if L>1 else 0.0
                pol_mean   = float(vals_pol.mean())
                pol_std    = float(vals_pol.std(unbiased=False)) if L>1 else 0.0
                vol_mean   = float(vals_vol.mean())
                vol_std    = float(vals_vol.std(unbiased=False)) if L>1 else 0.0
            else:
                gravy_mean = pol_mean = vol_mean = 0.0
                gravy_std = pol_std = vol_std = 0.0
            feat_g += [gravy_mean, gravy_std, pol_mean, pol_std, vol_mean, vol_std]

            # 电荷/酸碱
            net_q_pH7 = approx_net_charge(seq, ph=PH)
            charge_density = net_q_pH7 / L if L>0 else 0.0
            pI = estimate_pI(seq) if L>0 else 0.0
            feat_g += [net_q_pH7, charge_density, pI]

            # 质量/芳香/复杂度 + 脂肪族指数
            mw = peptide_mass(seq)
            entropy_comp = composition_entropy(cnt, L)
            feat_g += [mw, frac_aroma, entropy_comp]
            feat_g.append(aliphatic_index(seq))

            # 结构倾向（Chou–Fasman 简版）
            frac_cf_H = sum(1 for a in seq if a in CF_HELIX) / L if L>0 else 0.0
            frac_cf_E = sum(1 for a in seq if a in CF_STRAND) / L if L>0 else 0.0
            frac_cf_T = sum(1 for a in seq if a in CF_TURN) / L if L>0 else 0.0
            feat_g += [frac_cf_H, frac_cf_E, frac_cf_T]

            # 二肽（可选）
            if use_dipeptide:
                feat_g.extend(dipeptide_freq(seq).tolist())

            # ======= 局部/频域：与你原设计一致 =======
            feat_l = []
            for pname, table in props:
                x = map_prop(seq, table)          # (L,)
                xz = zscore(x)                    # 标准化
                bands, centroid = spectral_bands(xz, freq_bands=freq_bands)
                feat_l.extend(bands.tolist())
                feat_l.append(float(centroid))
                wmean, wmax = window_std_stats(xz, win=win, stride=stride)
                feat_l.extend([wmean, wmax])
                ac = short_autocorr(xz, max_lag=autocorr_max_lag)
                feat_l.extend(ac.tolist())

            global_feat_list.append(torch.tensor(feat_g, dtype=torch.float32))
            local_feat_list.append(torch.tensor(feat_l, dtype=torch.float32))

    # ------ 打包返回 ------
    global_feats = torch.stack(global_feat_list) if global_feat_list else torch.empty(0, len(global_names))
    local_feats  = torch.stack(local_feat_list)  if local_feat_list  else torch.empty(0, len(local_names))
    out = {
        "global_feats": global_feats,
        "local_feats":  local_feats,
        "global_names": global_names,
        "local_names":  local_names,
        "ids":          ids
    }
    return out

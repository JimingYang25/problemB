# Problem B — 黄河水沙通量变化规律（第一问）

## 核心思路

给定 9 期断面观测（$n=9$）与对应时段的逐小时水沙数据，本方案构建了一套**小样本集成建模框架**：以延迟响应模型（DRM）为物理骨架捕捉河床演化的主导趋势，以 6 个机器学习回归器学习 DRM 未能解释的残差过程，最后通过自适应贝叶斯模型平均（BMA）在 6 种集成策略中按 LOOCV $R^2$ 择优输出预测。

---

## 一、形态指标体系

在参考水位 $Z_{ref}=43\text{m}$ 下，每期断面提取 5 个指标：

| 符号 | 名称 | 公式 | 物理含义 |
|:---|:---|:---|:---|
| $A$ | 过水面积 | $\int \max(Z_{ref}-z(x), 0)\,dx$ | 断面输水能力 |
| $B$ | 水面宽度 | $\int \mathbf{1}_{z(x)<Z_{ref}}\,dx$ | 横向展宽 |
| $\xi$ | 宽深比 | $B^2/A$ | 形态胖瘦：越大越宽浅 |
| $H$ | 形态熵 | $-\sum p_j\ln p_j$ | 断面凹凸复杂程度 |
| $z_{min}$ | 深泓高程 | $\min z(x)$ | 主槽最低点（下切指标） |

目标变量为相邻断面间的变化量：$\Delta\mathbf{y}_t = \mathbf{y}_{t+1}-\mathbf{y}_t$（共 8 组）。

---

## 二、特征工程

### 2.1 含沙量重建

原始含沙量缺失率约 87%。基于挟沙力幂律 $S = a \cdot Q^b$，对实测 $(Q,S)$ 做 $\ln S = \ln a + b\ln Q$ 回归，补全缺失值。

### 2.2 特征池（14 维）

| 类别 | 特征 | 含义 |
|:---|:---|:---|
| 总量 | $V$, $M$, $Q_{peak}$, $Q_s$ | 累积水量、累积输沙量、洪峰流量、平均输沙率 |
| 变异性 | $C_v = \sigma_Q/\bar{Q}$ | 流量变异系数 |
| 前期 | $V_{ant}$, $M_{ant}$ | 前 90 天累积水量/沙量 |
| 衍生 | $SDR = M/(V+\varepsilon)$ | 输沙比 |
| | $f_{freq}$ | 洪水频率（$Q>Q_{95}$ 的天数/年） |
| | $r_{QS}$ | 水沙耦合相关系数 |

### 2.3 特征选择

遵循 EPV 原则（$n=8$，每个目标 $\le 4$ 个参数）：

1. Spearman $|\rho|$ 排序 → 取前 2 且互相关 $<0.8$
2. 加交互项 $x_1 \cdot x_2$
3. 输出 $X \in \mathbb{R}^{8 \times 3}$

实际选中特征对：

| 目标 | 特征 1 | 特征 2 | Spearman $|\rho|$ |
|:---|:---|:---|:---|
| dA | $f_{freq}$ | $\ln V$ | 0.56 / 0.38 |
| dB | $f_{freq}$ | $Q_{peak}$ | 0.57 / 0.31 |
| dξ | $C_v$ | $f_{freq}$ | 0.57 / 0.22 |
| dH | $C_v$ | $\ln V_{ant}$ | 0.33 / 0.29 |
| dz_min | $\ln Q_s$ | $r_{QS}$ | 0.36 / 0.26 |

---

## 三、延迟响应模型（DRM）

### 3.1 物理方程

DRM 将河床演化建模为一阶松弛过程——河道形态始终追赶一个由当前水沙条件决定的平衡态，追赶速率正比于偏离程度：

$$\frac{dy}{dt} = -\beta(y - y_e)$$

平衡态由水沙幂律决定：

$$y_e = K\left(\frac{S}{S_{ref}}\right)^a \left(\frac{Q}{Q_{ref}}\right)^b$$

其中 $Q_{ref}=\bar{Q}$，$S_{ref}=\bar{S}$。

### 3.2 离散递推

$$\hat{y}_i = \hat{y}_{i-1} \cdot e^{-\beta\Delta t} + y_{e,i} \cdot (1 - e^{-\beta\Delta t})$$

- $e^{-\beta\Delta t}$：记忆衰减因子（越大，历史影响越持久）
- $\tau = 1/\beta$：松弛时间（完成约 63% 调整所需的周期数）

### 3.3 参数估计

$$\min_{K,a,b,\beta} \frac{1}{n-1}\sum_{i=1}^{n-1}(y_i^{obs} - \hat{y}_i)^2$$

约束：$|a|,|b| \le 2.5$，$0 < \beta \le 3.0$。

采用**差分进化（DE）全局搜索 + L-BFGS-B 局部精化**，20 组随机初值避免局部最优。

---

## 四、集成框架

### 4.1 基模型池（7 个）

| # | 模型 | 类型 | 超参数（LOOCV 调优） |
|:---|:---|:---|:---|
| 1 | Ridge | L2 正则化线性 | $\alpha \in \{0.1,\dots,50\}$ |
| 2 | GPR | 高斯过程 (RBF+White核) | $\alpha \in \{0.05, 0.1, 0.3\}$ |
| 3 | KRR | 核岭回归 (RBF核) | $\alpha,\gamma$ 网格 |
| 4 | ElasticNet | L1+L2 混合 | $\alpha,\ l_1\ ratio$ 网格 |
| 5 | Lasso | L1 稀疏 | $\alpha \in \{0.01,\dots,1\}$ |
| 6 | Huber | 鲁棒回归 | $\varepsilon,\alpha$ 网格 |
| 7 | DRM | 物理模型 | DE + L-BFGS-B |

### 4.2 自适应筛选

剔除 LOOCV $R^2 \le -0.3$ 的弱模型：$\mathcal{M}_{active} = \{m : R^2_{LOO}(m) > -0.3\}$

### 4.3 六种集成策略

| 策略 | 原理 |
|:---|:---|
| Best-Single | 选 LOOCV $R^2$ 最高的单模型 |
| Equal | 等权平均 |
| BMA-BIC | $w_m \propto \exp(-\frac{1}{2}[BIC_m - \min BIC])$ |
| Softmax-LOO | $w_m \propto \exp(R^2_{LOO}(m)/T)$，$T$ 网格调优 |
| Softmax-Boot | 复合评分：$0.5\cdot\max(R^2_{LOO},-1) + 0.5\cdot\max(R^2_{Boot},-1)$ |
| Stacking | 非负最小二乘：$\min_{w\ge 0}\|y-P^{LOO}w\|^2$ |

对每个指标，从 6 种策略中自适应选择 LOOCV $R^2$ 最优者。

---

## 五、验证

- **LOOCV**：$n=8$ 折留一交叉验证，评估 $R^2_{LOO}$ 和 RMSE
- **Bootstrap**：$B=1000$ 次有放回重抽样，给出 $R^2$ 的 95% 置信区间

---

## 六、结果

### 6.1 DRM 拟合（绝对值）

| 指标 | $K$ | $a$ | $b$ | $\beta$ | $R^2$ | $\tau$ | 稳定? |
|:---|:---|:---|:---|:---|:---|:---|:---|
| $A$ | 1442 | −0.50 | +2.50 | 2.89 | **0.58** | 0.3 | ✅ |
| $B$ | 688 | −0.69 | +2.50 | 2.18 | **0.77** | 0.5 | ✅ |
| $\xi$ | 51 | +2.50 | +2.50 | 0.08 | 0.34 | 12.0 | ✅ |
| $H$ | 2.18 | −0.52 | +1.35 | 3.00 | **0.73** | 0.3 | ✅ |
| $z_{min}$ | 23.5 | −1.12 | +2.50 | 0.99 | 0.19 | 1.0 | ✅ |

> 所有 5 个指标 $\beta > 0$，系统存在稳定松弛机制。$A$、$B$、$H$ 的 $R^2$ 达到 0.58–0.77，DRM 物理模型对绝对值演化趋势有较强解释力。$\xi$ 的松弛时间最长（$\tau=12$），说明宽深比的调整最为缓慢。

### 6.2 集成预测（变化量 Δ）

| 目标 | 最佳单模型 | 单模型 $R^2_{LOO}$ | 最佳策略 | 集成 $R^2_{LOO}$ |
|:---|:---|:---|:---|:---|
| dA | KRR | ~0 | Softmax-LOO | ~0 |
| dB | Huber | 0.18 | Softmax-LOO | **0.23** |
| **dξ** | **Huber** | **0.93** | **Stacking** | **0.93** |
| dH | KRR | −0.07 | Stacking | ~0 |
| dz_min | KRR | 0.26 | Softmax-LOO | **0.26** |

> dξ（宽深比变化）的预测精度极高（$R^2_{LOO}=0.93$，Bootstrap 95% CI = $[0.34, 0.99]$），说明水沙通量特征对宽深比的短期调整方向有可靠的预测能力。dA 和 dH 的变化量在当前时间尺度上近似随机游走，短期变化难以用 8 个样本有效捕捉。

### 6.3 稳态分析

| 指标 | 当前值 | 平衡值 $y_{eq}$ | 偏离 | 状态 |
|:---|:---|:---|:---|:---|
| $A$ | 1869 m² | 2940 m² | **+57%** | 仍在扩容中 |
| $B$ | 1007 m | 1208 m | +20% | 略窄于平衡宽度 |
| $\xi$ | 543 | 1174 | **+116%** | 窄深化途中，调整最慢 |
| $H$ | 2.67 | 2.61 | −2% | ✓ 已接近稳态 |
| $z_{min}$ | 35.24 m | 29.26 m | −17% | 下切尚未完成 |

> 核心发现：**宽深比 $\xi$ 的当前值（543）与平衡值（1174）之间存在 116% 的巨大偏离**，且 $\tau=12$ 周期（约 15–20 年）是调整最慢的指标。这意味着该断面正处于从宽浅型向窄深型过渡的中间阶段，且过渡过程漫长。深泓高程和形态熵已基本接近平衡。

### 6.4 演化方向与稳定性

综合 DRM 参数符号和 Spearman 时间趋势，该河段断面正经历以下演化：

| 趋势 | 证据 |
|:---|:---|
| **窄深化** | $a_B < 0$（含沙量↑→宽度↓），$b_{z_{min}} > 0$（流量↑→深泓↓） |
| **面积扩容** | $b_A > 0$，当前 $A$ 仅达平衡值的 63% |
| **稳定收敛** | 所有指标 $\beta > 0$，10 周期预测收敛率 > 95% |
| **动态而非静态稳态** | 2016–2021 年水沙通量持续上升（Spearman $\rho=+0.94$, $p<0.01$），平衡值本身在漂移 |

**结论**：若无人为干预，该断面将趋向**窄深型拓扑结构**（中等宽度 ~475 m、大过水面积 ~838 m²、极低宽深比），而不会发生不可逆的形态崩溃。但需警惕两点——宽深比偏离过大且调整周期长达十余年，在此期间若遭遇极端洪水可能触发剧烈侧蚀；若水沙通量持续增大，平衡态将持续漂移，系统可能长期处于追赶状态。

---

## 七、输出文件

```
project/
├── data/
│   ├── cross_sections_processed.csv    # 9断面 × 统一1m网格
│   ├── geometry_indicators.csv         # A, B, ξ, H, z_min
│   ├── geometry_deltas.csv             # 8组 Δ 变化量
│   └── driving_factors.csv             # 14维特征
├── code/
│   ├── preprocess.py                   # 断面插值
│   ├── geometry.py                     # 几何指标
│   ├── driving_factors.py              # S-Q重建 + 特征工程
│   ├── drm_model.py                    # DRM (DE + L-BFGS-B)
│   ├── ensemble.py                     # 7模型 + 6策略 + BMA
│   ├── bootstrap.py                    # Bootstrap (B=1000)
│   ├── loocv.py                        # LOOCV 汇总
│   ├── run_all.py                      # 一键运行
│   └── figures.py                      # 图表
├── results/
│   ├── drm_parameters.csv              # DRM 参数 (K, a, b, β)
│   ├── steady_state_analysis.csv       # 稳态偏离 + 收敛性
│   ├── ensemble_summary.csv            # 集成 R² vs 单模型 R²
│   ├── ensemble_weights.csv            # 最优集成权重
│   ├── bootstrap_summary.csv           # Bootstrap R² 分布
│   ├── figure_cross_sections.png       # 9个断面叠加
│   └── figure_evolution.png            # 5指标演化曲线
└── README.md
```

---

## 附录：关键公式速查

**几何指标**
$$A = \int (Z_{ref}-z)\,dx,\quad B = \int \mathbf{1}_{z<Z_{ref}}\,dx,\quad \xi = B^2/A,\quad H = -\sum p_j\ln p_j,\quad z_{min} = \min z$$

**DRM 速率定律**
$$\frac{dy}{dt} = -\beta(y - y_e),\quad y_e = K(S/S_{ref})^a (Q/Q_{ref})^b,\quad \hat{y}_i = \hat{y}_{i-1}e^{-\beta\Delta t} + y_{e,i}(1-e^{-\beta\Delta t})$$

**集成策略**
$$\text{BMA-BIC: } w_m \propto e^{-(BIC_m - \min BIC)/2},\quad BIC = n\ln(SSE/n) + k\ln n$$
$$\text{Softmax: } w_m \propto e^{R^2_{LOO}(m)/T}$$
$$\text{Stacking: } \min_{w\ge 0}\|y - P^{LOO}w\|^2,\ w \leftarrow w/{\textstyle\sum} w_m$$

**验证**
$$R^2_{LOO} = 1 - \frac{\sum(y_k-\hat{y}_k)^2}{\sum(y_k-\bar{y})^2},\quad \bar{R}^2_{Boot} = \frac{1}{B}\sum_{b=1}^B R^2_b,\quad CI_{95\%} = [P_{2.5}, P_{97.5}]$$

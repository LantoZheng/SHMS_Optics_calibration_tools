# shms_optics_calibration 包详细技术文档

## 目录

1. [物理背景](#1-物理背景)
2. [包架构概览](#2-包架构概览)
3. [数据读取模块（data_io）](#3-数据读取模块data_io)
4. [数据预处理模块（preprocessing）](#4-数据预处理模块preprocessing)
5. [聚类算法模块（clustering）](#5-聚类算法模块clustering)
6. [标定模块（calibration）](#6-标定模块calibration)
7. [评估模块（evaluation）](#7-评估模块evaluation)
8. [可视化模块（visualization）](#8-可视化模块visualization)
9. [配置系统（config）](#9-配置系统config)
10. [参数选择对结果的影响](#10-参数选择对结果的影响)
11. [完整工作流程示例](#11-完整工作流程示例)
12. [参数速查表](#12-参数速查表)

---

## 1. 物理背景

### 1.1 SHMS 光学标定的必要性

SHMS（Super High Momentum Spectrometer，超高动量谱仪）是美国杰弗逊实验室（Jefferson Lab）Hall C 实验大厅的核心探测设备之一，设计用于 2.0–11.0 GeV/c 动量范围内的高精度电子散射实验。其磁铁构型为 **dQQQD**（一个水平偏转磁铁 + 三个四极磁铁 + 一个偶极磁铁）。

粒子在从靶到焦平面的飞行过程中，其轨迹由这些磁铁决定。**光学标定**的目的是精确确定从焦平面测量量（探测器直接测量）到靶面物理量（粒子散射顶点处的运动学量）之间的数学变换关系，即确定**重建矩阵**的各元素。

若标定不准确，重建的筛板孔位置会模糊或扭曲，导致所有物理分析结果产生系统误差。

### 1.2 坐标系定义

| 坐标系 | 变量 | 方向 | ROOT 分支名 |
|--------|------|------|-------------|
| 焦平面（Focal Plane） | $x_{fp}$ | 垂直（色散方向） | `P.dc.x_fp` |
| 焦平面 | $y_{fp}$ | 水平（非色散方向） | `P.dc.y_fp` |
| 焦平面 | $x'_{fp} = dx/dz$ | 垂直角度 | `P.dc.xp_fp` |
| 焦平面 | $y'_{fp} = dy/dz$ | 水平角度 | `P.dc.yp_fp` |
| 靶面（Target） | $\delta = dp/p$ | 动量偏差 | `P.gtr.dp` |
| 靶面 | $x'_{tar}$（θ） | 垂直角度 | `P.gtr.th` |
| 靶面 | $y'_{tar}$（φ） | 水平角度 | `P.gtr.ph` |
| 靶面 | $y_{tar}$ | 水平位置 | `P.gtr.y` |
| 靶面 | $z_{react}$ | 反应顶点 Z 位置 | `P.react.z` |

**重要约定**：对于 SHMS，$+y_{tar}$ 方向指向上游（朝向束流转储区方向）。

### 1.3 筛板方法（Sieve Slit Method）

光学标定的标准实验方法：

1. **数据采集**：在谱仪入口前方插入**筛板**（Sieve Slit）——一块重金属板，上面按精确几何图案钻出小孔阵列。使用碳箔靶采集数据。
2. **初始重建**：用现有矩阵元素（如来自 COSY 模拟）重放数据。若光学偏差，筛孔重建图像会模糊或形变。
3. **事例筛选**：识别属于各个筛孔的事例，确定每个孔的"真实"位置（由已知几何给出）。
4. **最小化（SVD）**：定义 $\chi^2$ 函数，用奇异值分解（SVD）最小化重建孔位置与真实孔位置之差，从而求得优化后的矩阵元素。

本软件包的核心任务正是**第 3 步**：从实验数据中自动识别各筛孔对应的事例群组（聚类），并建立其在靶面坐标系中的网格索引。

---

## 2. 包架构概览

```
shms_optics_calibration/
├── __init__.py        # 公共接口导出
├── config.py          # 所有配置参数与默认值
├── data_io.py         # ROOT 文件读取与靶面投影计算
├── preprocessing.py   # 箔片位置分类与数据准备
├── clustering.py      # 聚类算法（DBSCAN / HDBSCAN / 两段式）
├── calibration.py     # 网格索引构建与筛孔位置估算
├── evaluation.py      # 基准评估（效率/纯度/可分离性）
├── visualization.py   # 可视化函数
├── README.md          # 快速入门说明（英文）
└── DOCUMENTATION.md   # 本详细技术文档（中文）
```

### 典型工作流程

```
ROOT 文件
    │
    ▼ data_io.load_and_prepare_data()
    │  （加载数据 → 靶面投影 → 范围过滤）
    │
    ▼ preprocessing.classify_foils_with_range()
    │  （按 P_gtr_y 分布识别多箔片位置）
    │
    ▼（对每个箔片位置）
    │
    ▼ clustering.cluster_by_foil_position()
    │  可选算法：
    │  ├─ auto_dbscan_clustering()       # DBSCAN 自动参数搜索
    │  ├─ peel_and_cluster_edges()       # 边缘补充聚类（第二阶段）
    │  ├─ two_entry_dbscan()             # 两段式 DBSCAN（推荐）
    │  └─ auto_hdbscan_clustering()      # HDBSCAN 层次聚类
    │
    ▼ calibration.build_full_grid_index()
    │  （从聚类中心构建行/列网格索引）
    │
    ▼ evaluation.calculate_cluster_metrics()  （可选，需仿真数据）
       （效率 / 纯度 / 可分离性评估）
```

---

## 3. 数据读取模块（data_io）

### 3.1 `load_root_file()`

从 ROOT 文件读取数据到 pandas DataFrame。

**参数说明：**

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `file_path` | str | — | ROOT 文件路径 |
| `tree_name` | str | `"T"` | TTree 名称。Hall C 标准回放软件（hcana）生成的树通常命名为 `"T"` |
| `branches` | list\[str\] \| None | None | 要读取的分支名列表。`None` 表示读取所有分支（内存消耗大，建议指定） |
| `verbose` | bool | True | 是否打印文件路径、事例数等信息 |

**依赖**：需要安装 `uproot`（`pip install uproot`）。

**使用建议**：
- 对于大型 ROOT 文件（>10 GB），强烈建议通过 `branches` 参数只读取所需列，可将内存占用减少 10 倍以上。
- 如果文件包含向量分支（如多径迹数据），`uproot` 会将其转换为 `awkward` 数组；配合 `library="pd"` 选项会自动展开为多行。

---

### 3.2 `project_to_target()` 与 `add_target_projection()`

将重建的靶面变量投影到筛板平面坐标。

**投影公式（来自 SHMS 光学模型）：**

```
target_x = P_gtr_x + P_gtr_th × z_coefficient

target_y = (-0.019 × dp + 0.00019 × dp² + 213.0 × ph + y)
         + 40.0 × (-0.00052 × dp + 0.0000052 × dp² + ph)
```

其中 `z_coefficient = 253.0` 是筛板到靶面的距离（单位：cm）。

**`TargetProjectionConfig` 参数说明：**

| 参数 | 默认值 | 物理含义 |
|------|--------|---------|
| `x_z_coefficient` | 253.0 | 筛板到靶面距离（cm）。改变此值等效于修改筛板的 Z 位置假设，直接影响 `target_x` 的计算精度 |
| `y_dp_linear` | −0.019 | $\delta$（动量偏差）对 target_y 的一阶贡献系数（色散修正） |
| `y_dp_quadratic` | 0.00019 | $\delta$ 的二阶贡献系数（非线性色散修正） |
| `y_ph_coefficient` | 213.0 | $\phi$（水平角）贡献系数（= 138.0 + 75.0，来自 SHMS 光学几何） |
| `y_offset_dp_linear` | −0.00052 | 偏移项中 $\delta$ 的一阶系数 |
| `y_offset_dp_quadratic` | 0.0000052 | 偏移项中 $\delta$ 的二阶系数 |
| `y_offset_multiplier` | 40.0 | 偏移项的总放大系数（来自筛板偏移量几何） |

> **参数影响分析**：
> - `x_z_coefficient` 若偏差 1 cm，则 $x'_{tar} \sim 10^{-2}$ rad 量级的角度会产生约 2.5 mm 的 target_x 偏移——这对筛孔识别有显著影响。
> - `y_ph_coefficient` 的值 213.0 = 138.0 + 75.0，分别来自主弯转磁铁后的路径长度和修正量，此系数偏差会使整个 target_y 图案在水平方向整体平移。
> - 所有系数都经过对 SHMS 真实数据的仔细拟合，除非有明确的物理依据（例如使用不同磁场设置），否则不建议修改。

**`DataLoadingConfig` 参数说明：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `feature_cols` | `['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']` | 焦平面变量列名（用于 4D 可分离性分析） |
| `target_cols` | `['P_gtr_dp', 'P_gtr_th', 'P_gtr_ph', 'P_react_z']` | 靶面变量列名 |
| `target_x_range` | (−20.0, 20.0) | target_x 的有效范围（cm），超出范围的事例被过滤 |
| `target_y_range` | (−20.0, 20.0) | target_y 的有效范围（cm） |

---

### 3.3 `load_simulation_data()`

读取 COSY/SIMC 仿真 ROOT 文件，同时计算重建的筛板图案和真值信息。

**读取的仿真分支：**

| 分支名 | 对应物理量 |
|--------|-----------|
| `psdelta` | 动量偏差 $\delta$ |
| `psyptar` | 水平角 $\phi$ |
| `psxptar` | 垂直角 $\theta$ |
| `psytar` | 靶面水平位置 $y_{tar}$ |
| `fry` | 束流光栅 Y 位置 |
| `xsieve`, `ysieve` | 真值筛孔物理坐标（cm） |
| `xsnum`, `ysnum` | 真值筛孔编号（行/列索引） |

**输出的 `truth_hole_id`** 由 `xsnum * 100 + (ysnum % 100)` 计算，形成唯一的孔标识符，用于与聚类结果对比。

---

## 4. 数据预处理模块（preprocessing）

### 4.1 `classify_foils_with_range()`

根据 `P_gtr_y` 分布中的峰值，自动识别多箔片位置并分类事例。

**背景**：实验中通常使用多片碳箔靶（如 3 片，分别位于不同 Z 位置）来扩展靶面覆盖范围。每片箔产生的事例在 `P_gtr_y` 分布中对应一个峰。

**算法步骤：**

1. 对 `col_name` 列（默认 `P_gtr_y`）在 `y_range` 范围内建立直方图（`bins` 个箱）
2. 用 `scipy.signal.find_peaks` 找到峰值：要求峰高 ≥ 最大峰高 × `peak_height_fraction`，且相邻峰间距 ≥ `peak_distance` 个箱
3. 用 `peak_widths` 计算每个峰的半高全宽（FWHM），换算为标准差 $\sigma = \text{FWHM} / 2.355$
4. 将 `[峰中心 - sigma_factor × σ, 峰中心 + sigma_factor × σ]` 范围内的事例归类为该箔片

**`FoilClassificationConfig` 参数说明：**

| 参数 | 默认值 | 含义与影响 |
|------|--------|-----------|
| `col_name` | `'P_gtr_y'` | 用于分类的列名。标准 Hall C 分析中，`P_gtr_y` 对应靶面水平位置，不同 Z 位置的箔片在此变量上有明显分离 |
| `bins` | 50 | 直方图箱数。**太少**（如 20）会使峰形模糊，难以区分相近的箔片；**太多**（如 200）则在数据量不足时产生虚假峰 |
| `sigma_factor` | 2.5 | 分类宽度因子。含义：以 $N × \sigma$ 为半宽进行分类。**增大**此值（如 3.0）纳入更多尾部事例，但可能引入相邻箔片的污染；**减小**（如 1.5）则分类更纯但效率降低 |
| `y_range` | (−5.0, 5.0) | `P_gtr_y` 的有效范围（cm）。应根据实验箔片位置设置，超出范围的事例被标记为 `foil_position = -1` |
| `peak_height_fraction` | 0.05 | 最小峰高（相对于最高峰的比例）。用于过滤统计噪声产生的虚假小峰。若某箔片数据量极少（如端部箔片），需适当减小此值 |
| `peak_distance` | 10 | 相邻峰之间的最小间距（以直方图箱数计）。防止把一个宽峰识别为多个峰 |

> **参数影响分析（sigma_factor）**：
>
> | sigma_factor | 纳入的高斯分布概率 | 适用场景 |
> |---|---|---|
> | 1.0 | ~68% | 极严格筛选，适合箔片间距很小时 |
> | 2.0 | ~95% | 适中 |
> | 2.5（默认） | ~99% | 推荐，平衡效率与纯度 |
> | 3.0 | ~99.7% | 宽松，适合数据量少的情况 |
> | 3.5+ | >99.9% | 可能引入相邻箔片污染 |

---

## 5. 聚类算法模块（clustering）

本模块是整个包的核心，提供三类聚类算法，均以"筛孔图案识别"为目标：每个筛孔对应一个空间密集的事例簇。

### 5.1 DBSCAN 自动参数搜索（`auto_dbscan_clustering`）

**DBSCAN 算法原理：**

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）基于密度可达性定义簇：若一个点的 $\varepsilon$ 邻域内至少有 `min_samples` 个点，则该点为核心点；所有从核心点密度可达的点属于同一簇；不属于任何簇的点标记为噪声（`is_noise = True`）。

**核心优势**：
- 无需预先指定簇的数量（筛孔数量可能因遮挡而未知）
- 能识别任意形状的簇
- 自动将稀疏区域的点归类为噪声

**本函数的自动优化策略（网格搜索）：**

1. 生成 eps 候选值：在 `eps_range` 内均匀取 10 个点
2. 生成 min_samples 候选值：若未指定，以 `len(data)/1000` 为基准，取其 50%、75%、100%、125%、150% 共 5 个候选值
3. 对所有 `(eps, min_samples)` 组合运行 DBSCAN
4. 检查物理约束：
   - 每个簇的 x 向和 y 向尺寸均不超过 `max_cluster_size`
   - 相邻簇中心间距不小于 `distance_threshold`
5. 在满足约束且簇数量在 `target_clusters` 范围内的参数组合中，选择簇数量最接近目标中心值的组合

**`DBSCANConfig` 参数说明：**

| 参数 | 默认值 | 含义与影响 |
|------|--------|-----------|
| `x_col` | `'target_x'` | X 坐标列名 |
| `y_col` | `'target_y'` | Y 坐标列名 |
| `eps_range` | (0.01, 0.2) | eps 参数搜索范围（cm）。eps 定义邻域半径：**太小**→每个点单独成簇（过分割）；**太大**→相邻孔合并（欠分割）。筛孔直径约 0.4–0.8 cm，典型最优 eps 约 0.05–0.15 cm |
| `target_clusters` | (50, 70) | 目标簇数量范围。SHMS 筛板的孔数由其几何设计决定（通常约 63 个孔，7 列×9 行），但边缘孔可能不可见。建议根据实验筛板规格设置 |
| `min_samples` | None | 每个核心点的最小邻居数。None 时自动计算（数据量/1000）。**增大**：对噪声更鲁棒，但小孔（数据量少的孔）可能被归为噪声；**减小**：能识别更多孔，但对噪声敏感 |
| `max_iterations` | 10 | 最大搜索迭代次数（当前实现中为固定网格搜索，不直接使用此参数进行迭代） |
| `distance_threshold` | 1.0 | 簇中心间距约束（cm）。筛板孔间距约 1.27–2.54 cm，将此值设为孔间距的 50%–80% 较为合理。**太大**：强制相邻簇合并；**太小**：允许多个簇对应同一个孔 |
| `max_cluster_size` | 2.2 | 单个簇的最大尺寸（cm）。对应筛板孔的物理尺寸上限。设为孔径的 2–3 倍，避免把两个相邻孔的事例合并成一个过大的簇 |

---

### 5.2 边缘区域聚类（`peel_and_cluster_edges`）

**问题背景**：由于 SHMS 接受度的非均匀性（中心区域统计量远多于边缘），标准 DBSCAN 在搜索整体最优 eps 时，往往因边缘区域密度不足而将其归为噪声，导致边缘筛孔被漏识别。

**算法（凸包剥离法）：**

1. 计算已识别的核心簇所有点的凸包（Convex Hull）
2. 以 `radius` 为缓冲半径扩展凸包边界（`radius > 0` 时收缩边界，`radius < 0` 时扩张）
3. 提取凸包边界外部的"边缘点"
4. 对边缘点单独运行 DBSCAN（参数独立搜索）
5. 将新发现的边缘簇与原有核心簇合并，重新编号

**`EdgeClusteringConfig` 参数说明：**

| 参数 | 默认值 | 含义与影响 |
|------|--------|-----------|
| `radius_candidates` | [0.3, 0.5, 0.8, 1.0, 1.5] | 凸包扩展半径的候选值（cm）。**较小值**（0.3）：只剥离真正的边缘薄层，保留更多点参与边缘聚类；**较大值**（1.5）：剥离更厚的边缘带，适合边缘孔与核心孔距离较大的情况 |
| `eps_candidates` | [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20] | 边缘 DBSCAN 的 eps 候选值。边缘区域密度较低，通常最优 eps 比核心区域略大 |
| `target_new_clusters` | (5, 15) | 期望新发现的边缘簇数量范围。根据筛板边缘孔数量设置（通常 SHMS 筛板有 10–16 个边缘孔未被核心 DBSCAN 识别） |
| `distance_threshold` | 1.0 | 新边缘簇与已有核心簇之间的最小中心距离（cm）。防止边缘 DBSCAN 重新发现已存在的核心孔 |

---

### 5.3 两段式 DBSCAN（`two_entry_dbscan`）

这是推荐的标准工作流，将核心聚类和边缘聚类串联：

```python
# 第一阶段：核心区域 DBSCAN（高密度区）
df, eps_core, n_core = auto_dbscan_clustering(df, config=core_config)

# 第二阶段：边缘区域剥离与聚类（低密度区）
df, eps_edge, n_total = peel_and_cluster_edges(df, config=edge_config)
```

**返回值结构：**

```python
params = {
    'core_eps':      float,   # 核心区域最优 eps
    'edge_eps':      float,   # 边缘区域最优 eps
    'core_clusters': int,     # 核心区域簇数
    'edge_clusters': int      # 新增边缘簇数
}
```

---

### 5.4 HDBSCAN 层次聚类（`auto_hdbscan_clustering`）

**HDBSCAN 算法原理：**

HDBSCAN（Hierarchical DBSCAN）通过构建密度层次树（mst → condensed tree）来识别簇，相比 DBSCAN 的优势在于：
- 对 eps 参数不敏感（自动学习局部密度）
- 能处理**密度不均匀**的数据（如筛孔中心区域与边缘区域密度差异大）
- 通过 `cluster_selection_method` 控制簇的层次选择策略

**`HDBSCANConfig` 参数说明：**

| 参数 | 默认值 | 含义与影响 |
|------|--------|-----------|
| `min_cluster_size_range` | (30, 100) | `min_cluster_size` 的搜索范围。HDBSCAN 的 `min_cluster_size` 决定了被认为是"有意义的簇"的最小事例数。**增大**：忽略小簇（对应少数事例的筛孔），噪声更少；**减小**：识别更多小簇，但可能产生假簇 |
| `min_samples_range` | (30, 80) | `min_samples` 的搜索范围。该参数控制核心点的判定严格程度（类似 DBSCAN 的 `min_samples`），越大越保守 |
| `target_clusters` | (50, 70) | 目标簇数量范围，含义与 DBSCAN 相同 |
| `max_iterations` | 10 | 最大搜索迭代次数（当前为固定网格搜索） |
| `distance_threshold` | 1.0 | 簇中心最小间距约束（cm） |
| `max_cluster_size` | 2.2 | 单簇最大尺寸（cm） |
| `cluster_selection_method` | `'leaf'` | 簇选择策略。`'leaf'`：选择层次树最底层的叶节点作为簇（产生更多、更细分的簇，适合筛孔识别）；`'eom'`（超额质量法）：选择层次上稳定的区域（产生更少、更稳健的簇） |
| `metric` | `'euclidean'` | 距离度量。`'euclidean'`（欧氏距离）适合 cm 量级的物理坐标 |
| `alpha` | 1.0 | 控制如何将单链接分割转化为合并树。标准值为 1.0 |

> **DBSCAN vs HDBSCAN 选择指南：**
>
> | 情况 | 推荐 |
> |------|------|
> | 标准数据质量，密度较均匀 | DBSCAN 两段式（更快、更稳定） |
> | 边缘与中心密度差异极大（>10x） | HDBSCAN（自适应局部密度） |
> | 需要精确控制簇数量 | DBSCAN（参数更直观） |
> | 数据量极大（>100万事例/箔片） | HDBSCAN（内存效率更好） |

---

### 5.5 按箔片位置批量聚类（`cluster_by_foil_position`）

对每个箔片位置分别应用聚类算法，返回字典：`{foil_position: {'df': DataFrame, 'params': dict, 'n_clusters': int}}`。

**参数说明：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `method` | `'dbscan'` | 聚类方法。可选：`'dbscan'`、`'hdbscan'`、`'two_entry'`。当 `method='dbscan'` 且 `use_two_entry=True`（默认）时，自动使用两段式 DBSCAN |
| `use_two_entry` | True | 当 `method='dbscan'` 时，是否启用第二阶段边缘聚类 |
| `foil_col` | `'foil_position'` | 箔片位置列名 |

---

## 6. 标定模块（calibration）

### 6.1 `build_grid_index_from_centers()`

从聚类中心坐标推断行/列网格索引，是将"无序的聚类结果"转化为"有结构的筛孔阵列"的关键步骤。

**算法步骤：**

1. **提取聚类中心**：对每个簇取中心坐标的均值，过滤噪声点
2. **合并过近中心**（`merge_threshold`）：用层次聚类（average linkage）合并距离小于阈值的中心
3. **PCA 对齐**（可选，`use_pca_alignment`）：将点云主轴对齐到坐标轴，消除筛板微小倾斜
4. **估算网格间距**：用 k 近邻（k=5）距离的中位数作为网格间距
5. **确定原点**：取最接近点云质心的聚类中心作为 `(row=0, col=0)` 原点
6. **分配行/列索引**：
   ```python
   row = round((aligned_y - origin_y) / grid_spacing)
   col = round((aligned_x - origin_x) / grid_spacing)
   ```
7. **统计缺失位置**：枚举行/列范围内所有期望位置，与实际检测位置取差集

**`GridIndexConfig` 参数说明：**

| 参数 | 默认值 | 含义与影响 |
|------|--------|-----------|
| `cluster_col` | `'cluster'` | 聚类标签列名 |
| `x_col` | `'cluster_center_x'` | 聚类中心 X 坐标列名 |
| `y_col` | `'cluster_center_y'` | 聚类中心 Y 坐标列名 |
| `use_pca_alignment` | False | 是否使用 PCA 对齐。**启用时**：先将点云旋转到主轴方向再分配网格索引，适用于筛板有明显倾斜（>2°）的情况；**不启用**（默认）：假设筛板行列与坐标轴近似平行 |
| `merge_threshold` | 0.5 | 中心合并阈值（cm）。两个中心若距离 < 0.5 cm，则合并为一个。**太大**：可能将相邻的真实孔合并（对于孔间距 ~1 cm 的筛板，应设为孔间距的 30%–40%）；**太小**：单个孔可能被重复索引 |

> **PCA 对齐的影响**：
>
> 若筛板相对于坐标轴有约 5° 的旋转，不使用 PCA 会导致网格索引计算误差约 `sin(5°) × spacing ≈ 0.09 × spacing`；对于间距 1.27 cm 的筛板，误差约 1.1 mm，仍在容许范围内。但若倾斜角度更大（>10°），强烈建议启用 PCA 对齐。

---

### 6.2 `get_missing_holes()`

识别筛孔网格中的缺失位置。

**参数说明：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `only_internal` | True | 是否只返回凸包内部的缺失孔（排除边缘效应）。`True`：只报告真正被遮挡的孔（在已检测孔的凸包内部）；`False`：也报告网格矩形范围内的所有缺失位置（包括边缘外的推断位置）|

---

### 6.3 `estimate_hole_positions()`

根据网格参数（间距、原点、旋转角度）估算缺失孔的物理坐标：

```python
aligned_x = origin_x + col * grid_spacing
aligned_y = origin_y + row * grid_spacing

# 若使用了 PCA 旋转，反变换回原始坐标系
if rotation_angle != 0:
    est_x = aligned_x * cos(-θ) - aligned_y * sin(-θ)
    est_y = aligned_x * sin(-θ) + aligned_y * cos(-θ)
```

估算位置可用于：在原始散点图上叠加标记，帮助手动检查该区域的数据分布，判断孔是否真实缺失还是被错误归为噪声。

---

## 7. 评估模块（evaluation）

### 7.1 `calculate_cluster_metrics()`

使用仿真真值标签（`truth_hole_id`）计算聚类质量指标。

**指标定义：**

- **纯度（Purity）**：对某个簇 $C_k$，设其中占比最大的真值孔为 $h^*$，则
  $$\text{Purity}(C_k) = \frac{|C_k \cap h^*|}{|C_k|}$$
  纯度高（≈1.0）表示该簇确实对应单一筛孔。

- **效率（Efficiency）**：对某个真值孔 $H_j$，
  $$\text{Efficiency}(H_j) = \frac{\text{被正确分配到主导该孔簇的事例数}}{|H_j|}$$
  效率高（≈1.0）表示该孔的大部分事例被正确找回。

**`BenchmarkConfig` 参数说明：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `truth_col` | `'truth_hole_id'` | 真值孔标签列名（仿真数据中） |
| `cluster_col` | `'cluster'` | 聚类结果列名 |

---

### 7.2 `calculate_separability_metrics()`

在四维焦平面相空间（$x_{fp}, y_{fp}, x'_{fp}, y'_{fp}$）中评估簇的可分离性。

**计算的指标：**

| 指标 | 范围 | 解释 |
|------|------|------|
| Silhouette Score（轮廓系数） | [−1, +1] | 越大越好：+1 表示簇内紧密、簇间远离；0 表示簇在边界上；负值表示点被错误分配 |
| Davies-Bouldin Index（DB 指数） | [0, ∞) | 越小越好：低值表示簇内紧密且簇间分离 |
| Calinski-Harabasz Score（CH 分数） | [0, ∞) | 越大越好：簇间方差与簇内方差之比 |
| Separability Ratio（可分离比） | 任意 | min_inter_cluster_dist / intra_cluster_dist，>1 表示簇间距离大于簇内半径（良好分离）|

**`SeparabilityConfig` 参数说明：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `fp_cols` | `['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']` | 用于分析的焦平面变量列名 |
| `normalize` | True | 是否在计算距离前对特征归一化（StandardScaler）。**强烈推荐保持 True**：四个焦平面变量量纲不同（位置单位 cm，角度单位 rad），不归一化会导致位置变量主导距离计算 |

---

## 8. 可视化模块（visualization）

### 8.1 `visualize_dbscan_results()`

绘制聚类散点图，信号点按簇 ID 着色，噪声点为黑色，簇中心用白色十字标记。

**`VisualizationConfig` 参数说明：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `figsize` | (10, 10) | 图像尺寸（英寸） |
| `xlim` | (−20.0, 20.0) | X 轴范围（cm）。应与 `target_x_range` 一致 |
| `ylim` | (−20.0, 20.0) | Y 轴范围（cm） |
| `point_size` | 0.2 | 信号点大小（事例密度大时应设小以避免遮挡） |
| `noise_point_size` | 0.1 | 噪声点大小 |
| `noise_alpha` | 0.3 | 噪声点透明度 |
| `signal_alpha` | 0.6 | 信号点透明度 |
| `cmap` | `'nipy_spectral'` | 颜色映射。`nipy_spectral` 提供高对比度的多色映射，适合区分大量簇（60+ 簇时效果好） |
| `grid_alpha` | 0.3 | 网格线透明度 |

### 8.2 其他可视化函数

| 函数 | 用途 |
|------|------|
| `visualize_clustering_summary()` | 2×2 子图：聚类结果 + 簇中心分布 + 簇大小直方图 + 统计文字 |
| `visualize_clusters_in_focal_plane()` | 在焦平面变量投影上显示聚类结果（4 个子图：$x_{fp}$-$y_{fp}$、$x_{fp}$-$x'_{fp}$ 等） |
| `visualize_foil_classification()` | `P_gtr_y` 分布直方图，标注箔片分类范围 |
| `visualize_target_plane()` | 靶面坐标的 2D 热图 |
| `visualize_benchmark_comparison()` | 多算法指标对比条形图 |
| `plot_efficiency_map()` | 筛孔位置的效率热图（需要网格索引） |

---

## 9. 配置系统（config）

### 9.1 配置类层次结构

所有配置类均为 Python `dataclass`，支持直接实例化和修改：

```python
from shms_optics_calibration import DBSCANConfig

# 使用默认配置
config = DBSCANConfig()

# 自定义配置
config = DBSCANConfig(
    eps_range=(0.05, 0.25),
    target_clusters=(55, 75),
    distance_threshold=1.2,
    max_cluster_size=2.5
)
```

### 9.2 配置优先级

当函数同时接受独立参数和 `config` 对象时，**`config` 对象优先**（覆盖独立参数）：

```python
# config 会覆盖 eps_range 和 target_clusters
df, eps, n = auto_dbscan_clustering(
    df,
    eps_range=(0.01, 0.1),    # 被 config 忽略
    target_clusters=(40, 50), # 被 config 忽略
    config=DBSCANConfig(eps_range=(0.05, 0.2), target_clusters=(55, 65))
)
```

### 9.3 全局常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `RANDOM_SEED` | 42 | 随机种子，用于可复现性 |
| `FEATURE_COLS` | `['P_dc_x_fp', ...]` | 焦平面特征变量列表 |
| `TARGET_COLS` | `['P_gtr_dp', ...]` | 靶面目标变量列表 |

---

## 10. 参数选择对结果的影响

### 10.1 DBSCAN eps 参数的影响

eps 是 DBSCAN 最关键的参数，直接决定聚类粒度：

```
eps 过小（< 0.03 cm）：
  ↓ 邻域半径过小
  ↓ 大量正常事例被标记为噪声
  ↓ 每个筛孔可能被分裂为多个子簇
  结果：簇数 >> 筛孔数，纯度高但效率极低

eps 适中（0.05–0.15 cm，推荐）：
  ↓ 正确识别高密度筛孔区域
  ↓ 噪声事例（核外散射、误重建）被正确排除
  结果：簇数 ≈ 筛孔数，纯度和效率均高

eps 过大（> 0.2 cm）：
  ↓ 相邻筛孔被合并
  ↓ 噪声被纳入簇
  结果：簇数 << 筛孔数，效率虚高但纯度极低
```

**自动搜索的作用**：`auto_dbscan_clustering` 通过网格搜索找到满足物理约束（`max_cluster_size`、`distance_threshold`）且簇数在目标范围内的最优 eps，避免手动调参的试错过程。

### 10.2 target_clusters 的影响

```
target_clusters 设置偏低（如 40–50，而实际孔数为 63）：
  → 算法倾向于使用较大的 eps
  → 导致相邻孔合并，遗漏真实孔

target_clusters 设置偏高（如 80–100）：
  → 算法使用较小的 eps
  → 孔被过分割为子区域
  → 需要后处理合并

推荐策略：
  1. 查阅筛板几何文档确认孔数（如 63 孔）
  2. 考虑实验中哪些孔可能不在接受度内（通常边缘孔约 5–10 个）
  3. 设置 target_clusters = (可见孔数下界, 总孔数)
```

### 10.3 两段式 DBSCAN 的优势

单段 DBSCAN 在以下情况失效：
- **中心-边缘密度比 > 5:1**：中心孔数据量丰富，边缘孔数据量稀少
- 为识别边缘孔而放大 eps → 中心孔被合并

两段式策略：
1. 先用严格的 eps（适配高密度中心区）识别核心孔
2. 再对边缘点单独用较松的 eps 识别剩余边缘孔
3. 结果：中心孔和边缘孔都能被准确识别

### 10.4 HDBSCAN cluster_selection_method 的影响

```
'leaf'（叶节点选择，默认推荐）：
  → 在层次树最细粒度层面选择簇
  → 产生更多、更小、更紧凑的簇
  → 适合筛孔识别（物理上孔就是独立的小簇）

'eom'（超额质量法）：
  → 选择"稳定"的大尺度结构
  → 产生更少、更大的簇
  → 可能将相邻孔合并
  → 适合识别连续结构（不适合离散筛孔识别）
```

### 10.5 sigma_factor 对箔片分类的影响

以三片箔片为例（位置 ≈ -2.5, 0, +2.5 cm，峰宽 $\sigma \approx 0.5$ cm）：

```
sigma_factor = 1.5：
  箔片 0 的分类范围：[-3.25, -1.75]
  中间无分类区：[-1.75, -0.75]（事例被标记为 -1）
  → 严格但可能损失约 13% 的事例

sigma_factor = 2.5（默认）：
  箔片 0 的分类范围：[-3.75, -1.25]
  与箔片 1 的分类范围 [-1.25, +1.25] 刚好相接
  → 推荐

sigma_factor = 3.0：
  箔片 0 的分类范围：[-4.0, -1.0]
  与箔片 1 的范围 [-1.5, +1.5] 有 0.5 cm 重叠
  → 相邻箔片事例相互污染，导致聚类结果中出现两个箔片的孔混叠
```

### 10.6 merge_threshold 对网格索引的影响

```
merge_threshold = 0（不合并）：
  → 若同一孔被分割为两个子簇（eps 偏小时），会产生两个相距 <0.5 cm 的中心
  → 两个中心分配同一网格位置，造成索引冲突

merge_threshold = 0.5 cm（默认，约为孔间距的 40%）：
  → 合并真正过近的重复中心
  → 保留相邻真实孔（间距 ~1.27 cm）不被合并

merge_threshold = 1.5 cm（过大）：
  → 将实际孔间距约 1.27 cm 的相邻孔合并
  → 网格索引严重缺失
```

---

## 11. 完整工作流程示例

### 11.1 标准实验数据分析

```python
from shms_optics_calibration import (
    load_and_prepare_data,
    classify_foils_with_range,
    cluster_by_foil_position,
    build_full_grid_index,
    get_missing_holes,
    estimate_hole_positions,
    visualize_dbscan_results,
    DBSCANConfig,
    EdgeClusteringConfig,
    FoilClassificationConfig,
    TargetProjectionConfig,
)

# ---- 步骤 1：加载数据 ----
df = load_and_prepare_data(
    "run12345.root",
    tree_name="T",
    add_projection=True,
    filter_range=True
)
print(f"加载事例数：{len(df):,}")

# ---- 步骤 2：箔片分类 ----
foil_config = FoilClassificationConfig(
    bins=60,              # 使用更多箱改善峰值分辨率
    sigma_factor=2.5,     # 保持默认
    y_range=(-6.0, 6.0),  # 扩大范围以覆盖三箔片位置
    peak_distance=8       # 若箔片间距较小则适当减小
)
df = classify_foils_with_range(df, config=foil_config, plot=True)
print(f"识别到的箔片数：{df['foil_position'].nunique() - 1}")  # 排除 -1

# ---- 步骤 3：配置聚类参数 ----
# 核心区域 DBSCAN
core_config = DBSCANConfig(
    eps_range=(0.03, 0.18),
    target_clusters=(45, 65),    # 预期约 55 个核心孔可见
    distance_threshold=0.9,      # 略小于孔间距（~1.27 cm）的 70%
    max_cluster_size=2.0         # 孔径约 0.4 cm，3 倍余量
)

# 边缘区域补充聚类
edge_config = EdgeClusteringConfig(
    radius_candidates=[0.3, 0.6, 1.0],
    eps_candidates=[0.06, 0.10, 0.15, 0.20],
    target_new_clusters=(5, 12)
)

# ---- 步骤 4：按箔片聚类 ----
results = cluster_by_foil_position(
    df,
    method='two_entry',       # 推荐使用两段式
    dbscan_config=core_config,
    edge_config=edge_config,
    verbose=True
)

# ---- 步骤 5：网格索引 ----
from shms_optics_calibration import GridIndexConfig

grid_config = GridIndexConfig(
    use_pca_alignment=False,  # 筛板无明显倾斜时保持 False
    merge_threshold=0.4       # 孔间距约 1.27 cm，取 ~30%
)
full_index, grid_params = build_full_grid_index(
    results,
    config=grid_config
)

# ---- 步骤 6：检查缺失孔 ----
for foil_pos, result in results.items():
    df_foil = result['df']
    centers = full_index[full_index['foil_position'] == foil_pos]
    params = grid_params[foil_pos]

    missing = get_missing_holes(centers, params, only_internal=True)
    if missing:
        estimated = estimate_hole_positions(centers, params, missing)
        print(f"箔片 {foil_pos}：{len(missing)} 个内部缺失孔")
        print(estimated[['row', 'col', 'estimated_x', 'estimated_y']])

# ---- 步骤 7：可视化 ----
for foil_pos, result in results.items():
    fig = visualize_dbscan_results(
        result['df'],
        n_clusters=result['n_clusters'],
        title_prefix=f"箔片 {foil_pos}",
        show=True
    )
```

### 11.2 仿真数据基准测试

```python
from shms_optics_calibration import (
    load_simulation_data,
    auto_dbscan_clustering,
    auto_hdbscan_clustering,
    two_entry_dbscan,
    calculate_cluster_metrics,
    calculate_separability_metrics,
    compare_algorithms,
)

# 加载仿真数据（含真值标签）
df_sim = load_simulation_data("simc_output.root")
print(f"仿真事例数：{len(df_sim):,}")
print(f"真值孔数：{df_sim['truth_hole_id'].nunique()}")

# 运行三种算法对比
results_comparison = {}

# 方法一：DBSCAN
df_db = df_sim.copy()
df_db, eps, n = auto_dbscan_clustering(df_db, target_clusters=(55, 70))
_, _, overall_db = calculate_cluster_metrics(df_db)
results_comparison['DBSCAN'] = {
    'df': df_db,
    'cluster_metrics': _,
    'truth_metrics': _,
    'overall': overall_db
}

# 方法二：两段式 DBSCAN
df_2e = df_sim.copy()
df_2e, params_2e, n_2e = two_entry_dbscan(df_2e)
cm_2e, tm_2e, overall_2e = calculate_cluster_metrics(df_2e)
results_comparison['Two-Entry DBSCAN'] = {
    'cluster_metrics': cm_2e,
    'truth_metrics': tm_2e,
    'overall': overall_2e
}

# 方法三：HDBSCAN
df_hd = df_sim.copy()
df_hd, params_hd, n_hd = auto_hdbscan_clustering(
    df_hd,
    min_cluster_size_range=(30, 80),
    target_clusters=(55, 70)
)
cm_hd, tm_hd, overall_hd = calculate_cluster_metrics(df_hd)
results_comparison['HDBSCAN'] = {
    'cluster_metrics': cm_hd,
    'truth_metrics': tm_hd,
    'overall': overall_hd
}

# 生成对比表
from shms_optics_calibration import compare_algorithms
comparison_table = compare_algorithms(results_comparison)
print(comparison_table.to_string(index=False))

# 4D 焦平面可分离性分析
global_metrics, per_cluster = calculate_separability_metrics(
    df_2e,
    normalize=True  # 必须归一化（量纲不同）
)
print(f"Silhouette Score: {global_metrics['silhouette_score']:.4f}")
print(f"Mean Separability Ratio: {global_metrics['mean_separability_ratio']:.3f}")
```

---

## 12. 参数速查表

### 12.1 所有配置类参数汇总

#### `DataLoadingConfig`

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `feature_cols` | `['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']` | list | 焦平面特征变量 |
| `target_cols` | `['P_gtr_dp', 'P_gtr_th', 'P_gtr_ph', 'P_react_z']` | list | 靶面目标变量 |
| `target_x_range` | (−20.0, 20.0) | tuple | target_x 过滤范围（cm） |
| `target_y_range` | (−20.0, 20.0) | tuple | target_y 过滤范围（cm） |

#### `TargetProjectionConfig`

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `x_z_coefficient` | 253.0 | float | 筛板-靶面 Z 距离（cm） |
| `y_dp_linear` | −0.019 | float | δ 线性色散修正系数 |
| `y_dp_quadratic` | 0.00019 | float | δ 二阶色散修正系数 |
| `y_ph_coefficient` | 213.0 | float | φ 贡献系数 |
| `y_offset_dp_linear` | −0.00052 | float | 偏移项 δ 线性系数 |
| `y_offset_dp_quadratic` | 0.0000052 | float | 偏移项 δ 二阶系数 |
| `y_offset_multiplier` | 40.0 | float | 偏移项放大系数 |

#### `FoilClassificationConfig`

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `col_name` | `'P_gtr_y'` | str | 分类依据列名 |
| `bins` | 50 | int | 直方图箱数 |
| `sigma_factor` | 2.5 | float | 分类半宽因子（× σ） |
| `y_range` | (−5.0, 5.0) | tuple | 有效值范围（cm） |
| `peak_height_fraction` | 0.05 | float | 最小相对峰高 |
| `peak_distance` | 10 | int | 相邻峰最小间距（箱数） |

#### `DBSCANConfig`

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `x_col` | `'target_x'` | str | X 坐标列名 |
| `y_col` | `'target_y'` | str | Y 坐标列名 |
| `eps_range` | (0.01, 0.2) | tuple | eps 搜索范围（cm） |
| `target_clusters` | (50, 70) | tuple | 目标簇数量范围 |
| `min_samples` | None | int\|None | 最小邻居数（None=自动） |
| `max_iterations` | 10 | int | 最大搜索次数 |
| `distance_threshold` | 1.0 | float | 簇中心最小间距（cm） |
| `max_cluster_size` | 2.2 | float | 单簇最大尺寸（cm） |

#### `EdgeClusteringConfig`

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `radius_candidates` | [0.3, 0.5, 0.8, 1.0, 1.5] | list | 凸包缓冲半径候选值（cm） |
| `eps_candidates` | [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20] | list | 边缘 eps 候选值（cm） |
| `target_new_clusters` | (5, 15) | tuple | 新增边缘簇数量目标范围 |
| `distance_threshold` | 1.0 | float | 新簇与已有簇的最小间距（cm） |

#### `HDBSCANConfig`

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `x_col` | `'target_x'` | str | X 坐标列名 |
| `y_col` | `'target_y'` | str | Y 坐标列名 |
| `min_cluster_size_range` | (30, 100) | tuple | min_cluster_size 搜索范围 |
| `min_samples_range` | (30, 80) | tuple\|None | min_samples 搜索范围（None=HDBSCAN 默认） |
| `target_clusters` | (50, 70) | tuple | 目标簇数量范围 |
| `max_iterations` | 10 | int | 最大搜索次数 |
| `distance_threshold` | 1.0 | float | 簇中心最小间距（cm） |
| `max_cluster_size` | 2.2 | float | 单簇最大尺寸（cm） |
| `cluster_selection_method` | `'leaf'` | str | 簇选择方法（`'leaf'` 或 `'eom'`） |
| `metric` | `'euclidean'` | str | 距离度量 |
| `alpha` | 1.0 | float | HDBSCAN alpha 参数 |

#### `SoftWeightedDBSCANConfig`（扩展功能）

> 此配置类用于软加权分区 DBSCAN，通过对不同 `P_gtr_y` 区域的点施加不同权重，在全局单次聚类中同时处理所有箔片。目前为实验性功能。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gtry_col` | `'P_gtr_y'` | 用于区域划分的列名 |
| `gtry_ranges` | `{'-2.5±1': (-3.5,-1.5), '0±1': (-1,1), '+2.5±1': (1.5,3.5)}` | 三个箔片对应的 P_gtr_y 区域 |
| `suppression_strength_grid` | (0.7, 0.8, 0.9) | 区域抑制强度搜索网格 |
| `sigma_grid` | (0.8, 1.0, 1.2) | 高斯权重 σ 搜索网格（cm） |
| `active_threshold_grid` | (0.3, 0.4, 0.5) | 激活权重阈值搜索网格 |
| `eps_grid` | (0.05, 0.08, ..., 0.20) | eps 搜索网格（cm） |
| `min_samples_grid` | (30, 50, 70, 90) | min_samples 搜索网格 |
| `assign_radius` | 0.8 | 将点分配给新簇的搜索半径（cm） |

#### `GridIndexConfig`

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `cluster_col` | `'cluster'` | str | 聚类标签列名 |
| `x_col` | `'cluster_center_x'` | str | 中心 X 列名 |
| `y_col` | `'cluster_center_y'` | str | 中心 Y 列名 |
| `use_pca_alignment` | False | bool | 是否启用 PCA 轴对齐 |
| `merge_threshold` | 0.5 | float | 中心合并阈值（cm） |

#### `VisualizationConfig`

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `figsize` | (10, 10) | tuple | 图像尺寸（英寸） |
| `xlim` | (−20.0, 20.0) | tuple | X 轴范围（cm） |
| `ylim` | (−20.0, 20.0) | tuple | Y 轴范围（cm） |
| `point_size` | 0.2 | float | 信号点大小 |
| `noise_point_size` | 0.1 | float | 噪声点大小 |
| `noise_alpha` | 0.3 | float | 噪声点透明度 |
| `signal_alpha` | 0.6 | float | 信号点透明度 |
| `cmap` | `'nipy_spectral'` | str | 颜色映射名称 |
| `grid_alpha` | 0.3 | float | 网格线透明度 |

#### `BenchmarkConfig`

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `truth_col` | `'truth_hole_id'` | str | 真值标签列名 |
| `cluster_col` | `'cluster'` | str | 聚类结果列名 |

#### `SeparabilityConfig`

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `fp_cols` | `['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']` | list | 焦平面特征变量 |
| `normalize` | True | bool | 是否归一化特征 |

---

### 12.2 推荐参数设置（按使用场景）

#### 场景 A：标准 SHMS 筛板（~63 孔，孔间距 ~1.27 cm）

```python
DBSCANConfig(
    eps_range=(0.03, 0.18),
    target_clusters=(50, 68),
    distance_threshold=0.9,
    max_cluster_size=2.0
)
EdgeClusteringConfig(
    radius_candidates=[0.3, 0.5, 0.8],
    target_new_clusters=(5, 15)
)
FoilClassificationConfig(
    sigma_factor=2.5,
    y_range=(-5.0, 5.0)
)
```

#### 场景 B：数据量较少（每孔 < 100 事例）

```python
DBSCANConfig(
    min_samples=5,           # 降低最小邻居数要求
    eps_range=(0.05, 0.25),  # 允许更大的邻域
    target_clusters=(40, 65)
)
HDBSCANConfig(              # 或改用 HDBSCAN
    min_cluster_size_range=(5, 30),
    cluster_selection_method='leaf'
)
```

#### 场景 C：数据量极大（每孔 > 10000 事例）

```python
DBSCANConfig(
    min_samples=100,         # 提高核心点要求，过滤边缘波动
    eps_range=(0.03, 0.12),  # 收紧搜索范围加快计算
    max_cluster_size=1.8     # 更严格的尺寸约束
)
```

#### 场景 D：筛板有明显倾斜（>5°）

```python
GridIndexConfig(
    use_pca_alignment=True,
    merge_threshold=0.4
)
```

---

## 参考文献

1. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD, 226–231.
2. Campello, R. J., Moulavi, D., & Sander, J. (2013). *Density-based clustering based on hierarchical density estimates*. PAKDD, 160–172.
3. Jefferson Lab Hall C: https://www.jlab.org/physics/hall-c
4. SHMS optics references: https://github.com/hszumila/SHMS_optics
5. JLab Hall C hcana analysis software: https://github.com/JeffersonLab/hcana

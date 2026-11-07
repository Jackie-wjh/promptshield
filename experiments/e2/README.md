# E2 BiLSTM 实验说明与复现指南

本目录包含了 **PromptShield** 项目中 **E2 实验（双向 LSTM 词向量初始化受控对比与错误案例分析）** 的完整源代码、执行脚本与评估逻辑。

---

## 1. 实验目标与设计

E2 的核心目标是评估循环神经网络（BiLSTM）在提示注入（Prompt Injection）检测任务中的表现，并深入探索**词向量初始化对模型性能与泛化性的影响**。

### 受控对照设置
- **对照组 A（Random）**：正态分布随机初始化的词向量（100 维）。
- **对照组 B（GloVe）**：预训练斯坦福 `glove.6B.100d` 词向量初始化（100 维）。
- **完全锁定的共有参数**：
  - 双向 LSTM（每个方向 128 维，双向拼接后输出 256 维）
  - Masked Mean Pooling（掩码均值池化，排除 padding 填充对句向量的稀释）
  - Dropout 0.3
  - 词表**严格仅从 D1 Train 提取**（共 9,386 词，GloVe 覆盖率 92.94%，OOV 率 7.06%）
  - Adam 优化器，学习率固定 $1 \times 10^{-3}$，批大小 64，最大长度 256
  - 最多训练 8 Epochs，仅根据 D1 Validation 在默认阈值 0.5 下的 Macro-F1 早停（Patience = 2）

---

## 2. 目录与文件架构

```text
experiments/e2/
├── __init__.py           # 包声明文件
├── model.py              # BiLSTM 神经网络结构定义（含 Masked Mean Pooling 与二分类头）
├── dataset.py            # 词表构建、GloVe 向量加载、PyTorch Dataset 与动态 Padding
├── run_e2.py             # 核心实验入口：全流程训练、早停选择、生成交接 scores.parquet 及曲线
├── export_evidence.py    # 辅助脚本：生成跨域泛化柱状图与统一模型指标表
├── analyze_cases.py      # 辅助脚本：从测试推断中提取 6 个典型 Bad Cases 并生成案例表
└── README.md             # 本说明文档
```

---

## 3. 环境依赖与运行复现

### 3.1 激活虚拟环境并安装依赖
在项目根目录下确保已激活 `NLP` 虚拟环境，并安装基础依赖：
```bash
pip install torch pandas pyarrow scikit-learn matplotlib
```

### 3.2 准备 GloVe 词向量（可选，内置自动下载）
`run_e2.py` 内置了从 HuggingFace 自动下载并解压 `glove.6B.100d.txt` 到 `data/pretrained/` 的功能。  
若离线运行，可手动将 `glove.6B.100d.txt` 放入 `data/pretrained/` 文件夹下。

### 3.3 一键运行实验
在项目根目录 `D:\NTUDevelop\6405\promptshield` 下依次执行以下命令：

#### 步骤 1：运行主实验（训练、早停、生成评分）
```bash
python -m experiments.e2.run_e2
```
*该脚本会在几分钟内完成两组初始化训练，自动保存最佳 Checkpoint，并在所有验证与测试集上生成 Raw Scores。*

#### 步骤 2：生成补充图表与模型汇总表
```bash
python -m experiments.e2.export_evidence
```
*自动生成跨域迁移柱状图 `e2_transfer_bar.png` 与 `selected_models.csv`。*

#### 步骤 3：提取 6 个典型错误案例
```bash
python -m experiments.e2.analyze_cases
```
*自动提取 3 个假阳性（FP）与 3 个假阴性（FN）错误样本，输出至 `results/e2/error_cases.csv`。*

---

## 4. 产出与交付物清单 (Artifacts)

运行完成后，所有交付成果均已按规范落地：

### 模型权重（供 M3 前端 Demo 与组内复现直接加载）
- `models/e2/bilstm_random_best.pt`：Random 条件最佳权重（4.89 MB，选定主模型）
- `models/e2/bilstm_glove_best.pt`：GloVe 条件最佳权重（4.89 MB）

### 交付给下游 M6 / M5 的交接包
- `results/e2/handoff/bilstm_scores.parquet`：包含 D1 Val (1,985行)、D1 Test (2,101行)、D2 Test (116行)、D3 Test (545行)，**共计 4,747 行无一缺失**，`transform_id` 统一对齐为 `"clean"`。
- `results/e2/handoff/selected_config.json`：最佳模型参数配置、延迟与显存元数据。
- `results/e2/handoff/run_metadata.json`：运行环境与测试时间戳。

### 调优记录与图表
- `results/e2/tuning.csv`：两组条件各 Epoch 的 Train/Val Loss 与 Macro-F1 详细过程记录。
- `results/e2/selected_models.csv`：模型 ID、参数量、单条推理延迟（中位数 0.54ms，P95 0.85ms）。
- `results/e2/figures/e2_learning_curves.png`：两组条件训练/验证曲线对比图。
- `results/e2/figures/e2_transfer_bar.png`：跨数据集泛化性能断崖式下跌直观柱状图。

### 报告与答辩支撑材料
- `results/e2/findings_and_limitations.md`：358 英文词的严谨学术分析，解释了 GloVe 在本任务中的领域偏差先验及跨域迁移局限。
- `results/e2/error_cases.csv`：6 个 Bad Cases 详细对照表。
- `results/e2/error_analysis.md`：深入剖析 6 个错误案例产生机制的专题文档。

---

## 5. 核心实验结论速记（答辩参考）

1. **Random 反超 GloVe**：
   Random 初始化（Val F1: 0.9869）略优于 GloVe 初始化（Val F1: 0.9848）。原因在于通用新闻/百科语料训练的 GloVe 将攻击指令词（如 `ignore`, `override`）视为中性词，模型需要耗费额外梯度调整先验；而 Random 能更纯粹直接地拟合 1.1 万条训练集中的领域专用对抗模式。
2. **跨域泛化断崖式退化**：
   - D1 In-Domain: 0.9619
   - D2 Out-of-Domain: 0.8448 (-11.7%)
   - D3 Complex Agentic: 0.7151 (-24.7%)  
   证明传统 BiLSTM 极度依赖表面词汇共现，面对复杂伪装与长程智能体指令时漏检率显著上升，凸显了后续引入 Transformer（DistilBERT/RoBERTa）的必要性。

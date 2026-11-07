# PromptShield 实验交接须知

本目录是供 E2–E6 组员开展实验的最小交接包。数据已完成冻结、清洗、稳定 ID 分配和完整性核验；E1 的三个 TF-IDF + Logistic Regression 模型也已冻结。组员应直接使用这里的 `data/processed` 文件，不需要重新下载或重新清洗数据。

## 1. 共同规则

1. 所有模型只使用 D1 Train 训练，使用 D1 Validation 选择超参数或 checkpoint。
2. 配置冻结后，才在 D1 Test、D2 Test 和 D3 Test 上进行最终评测。
3. D2、D3 只作为外部测试集；不得使用其训练集或测试标签调整模型。
4. D4 是 339 条全正常的 NotInject 压力测试数据，只用于冻结模型和阈值后的误报分析。
5. 所有成员必须保留原始 `sample_id`，不得重新随机划分数据。
6. 模型分数均不应描述为经过校准的概率。

## 2. 环境安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 足以读取 Parquet、加载 E1 模型并开展 E5/E6 的传统机器学习分析。E2–E4 还需要自行固定并记录 PyTorch、Transformers、Datasets 或 Accelerate 等神经网络依赖版本。

## 3. 数据文件

| 文件 | 行数 | 用途 | 使用者 |
|---|---:|---|---|
| `data/processed/d1_train.parquet` | 11,046 | 唯一模型训练来源 | E2、E3、E4、E5 |
| `data/processed/d1_validation.parquet` | 1,985 | 超参数、checkpoint 和 E6 阈值选择 | E2–E6 |
| `data/processed/d1_test.parquet` | 2,101 | 冻结配置后的同源测试 | E2–E6 |
| `data/processed/d2_test.parquet` | 116 | 冻结配置后的外部迁移测试 | E2–E6 |
| `data/processed/d3_test.parquet` | 545 | 冻结配置后的 Agentic 外部测试 | E2–E6 |
| `data/processed/d4_test.parquet` | 339 | 全正常文本的误报压力测试 | E5、E6 |

训练数据的核心字段为 `text` 和 `label`。`label=0` 表示正常，`label=1` 表示攻击。所有文件还包含稳定的 `sample_id` 和文本哈希。D3 的 `attack_family`、`technique`、`severity`、`language` 等字段仅用于分析，不得作为模型输入。

Parquet 可用 pandas 读取：

```python
import pandas as pd

df = pd.read_parquet("data/processed/d1_train.parquet")
print(df.shape)
print(df[["sample_id", "text", "label"]].head())
```

## 4. 数据版本和完整性

| 文件 | 用途 |
|---|---|
| `data/audit/manifest.json` | 固定数据集 revision、文件行数和 SHA-256 |
| `data/audit/integrity_report.json` | 已完成的数据映射、标签、清洗和精确重叠核验结果 |
| `configs/data_sources.json` | 数据来源、许可、角色和预期计数 |

`integrity_report.json` 的状态为 `passed`。该交接包不包含 raw 数据和逐行清洗决策，因为组员开展模型实验只需使用已经冻结的 processed 数据。

## 5. 分阶段使用测试集

### 阶段 A：训练与模型选择

E2–E4 首先只使用：

- `data/processed/d1_train.parquet`
- `data/processed/d1_validation.parquet`
- `data/audit/manifest.json`
- `data/audit/integrity_report.json`
- `configs/data_sources.json`

### 阶段 B：配置冻结后的评测

确定模型结构、超参数和 checkpoint 后，再使用：

- `data/processed/d1_test.parquet`
- `data/processed/d2_test.parquet`
- `data/processed/d3_test.parquet`

D4 不用于模型选择。E5 和 E6 应在模型及阈值冻结后才使用 `data/processed/d4_test.parquet`。

## 6. E1 冻结基线

E1 比较了三个使用同一 Logistic Regression 设置的管线：

- Word TF-IDF：1–2 grams，最多 50,000 个特征
- Char TF-IDF：3–5 grams，最多 50,000 个特征
- Combined TF-IDF：Word 与 Char 拼接，最多 100,000 个特征

三条管线均在固定阈值 0.5 下，根据 D1 Validation Macro F1 从 `C ∈ {0.1, 1, 10}` 中选择 C；最终均选择 `C=1`。这是 E1 的模型选择结果。`tF1` 和 `t5` 的正式选择、比较和解释属于 E6。

| 文件 | 用途 | 使用者 |
|---|---|---|
| `configs/e1.json` | E1 特征和分类器配置 | E5、E6、应用成员 |
| `models/e1/word.joblib` | 冻结的 Word TF-IDF LR 管线 | E6、应用成员 |
| `models/e1/char.joblib` | 冻结的 Char TF-IDF LR 管线 | E6、应用成员 |
| `models/e1/combined.joblib` | 冻结的 Combined TF-IDF LR 管线 | E5、E6、应用成员 |
| `results/e1/selected_models.csv` | C、验证 F1、特征数、模型大小和延迟 | E5、E6、结果整合成员 |
| `results/e1/evidence/` | tuning、特征维度、系数/局部解释、图、运行摘要和复算记录 | 报告整合与结果核验成员 |
| `results/e1/reference_only/` | E1 阶段曾计算的阈值和测试指标，仅供追溯 | E6 核验参考；不得直接作为最终小组结果 |
| `experiments/e1/run_e1.py` | E1 实现和评分函数参考 | E5、E6 |
| `experiments/e1/verify_e1.py` | 完整 E1 结果的复算逻辑参考；最小交接包未包含全部 E1 输出，不能直接完成全量 verify | E6 |
| `experiments/e1/explore_tuning.py` | E1 的 post hoc 探索脚本 | 非主实验；默认不用 |

`results/e1/reference_only` 中的阈值和指标是 E1 运行阶段留下的历史复算结果。E6 必须从 `results/e1/handoff/*_scores.parquet` 重新统一生成正式的 `thresholds.csv`、`metrics.csv` 和预测文件；最终报告采用 E6 的版本。

加载模型示例：

```python
import joblib
import pandas as pd

model = joblib.load("models/e1/combined.joblib")
validation = pd.read_parquet("data/processed/d1_validation.parquet")
score_attack = model.predict_proba(validation["text"].astype(str))[:, 1]
```

## 7. 各成员需要的文件

### E2：BiLSTM 初始化对照

开始训练：

- `data/processed/d1_train.parquet`
- `data/processed/d1_validation.parquet`
- `data/audit/manifest.json`
- `configs/data_sources.json`

配置冻结后再使用 D1、D2、D3 Test。词表只能由 D1 Train 建立。E2 需自行保存两种初始化的配置、学习曲线、OOV 统计、选定 checkpoint 和统一格式的预测文件。

### E3：DistilBERT 学习率对照

开始训练：

- `data/processed/d1_train.parquet`
- `data/processed/d1_validation.parquet`
- `data/audit/manifest.json`
- `configs/data_sources.json`

配置冻结后再使用 D1、D2、D3 Test。E3 需记录模型 revision、tokenizer、截断率、学习率、checkpoint、随机种子和运行环境。

### E4：RoBERTa 学习率与成本对照

所需公共文件与 E3 相同。E4 还应记录 batch-1 latency、P95 latency、模型大小、显存使用和截断率，并输出统一格式的预测文件。

### E5：受控 typo augmentation

需要：

- `data/processed/d1_train.parquet`
- `data/processed/d1_validation.parquet`
- `data/processed/d1_test.parquet`
- `data/processed/d2_test.parquet`
- `data/processed/d3_test.parquet`
- `data/processed/d4_test.parquet`
- `models/e1/combined.joblib`
- `configs/e1.json`
- `results/e1/selected_models.csv`
- `experiments/e1/run_e1.py`

E5 应复用冻结的 Combined 词表、IDF、特征配置和 `C=1`，只改变训练增强。Clean 与 Augmented 两个条件必须使用相同评测 ID，并保留 `parent_id` 和 `transform_id`。

### E6：阈值与误报分析

需要：

- `models/e1/word.joblib`
- `models/e1/char.joblib`
- `models/e1/combined.joblib`
- `results/e1/handoff/word_scores.parquet`
- `results/e1/handoff/char_scores.parquet`
- `results/e1/handoff/combined_scores.parquet`
- `data/processed/d1_validation.parquet`
- `data/processed/d1_test.parquet`
- `data/processed/d2_test.parquet`
- `data/processed/d3_test.parquet`
- `data/processed/d4_test.parquet`
- `results/e1/predictions/word_predictions.parquet`
- `results/e1/predictions/char_predictions.parquet`
- `results/e1/predictions/combined_predictions.parquet`
- `results/e1/selected_models.csv`
- `configs/e1.json`

`results/e1/handoff/*_scores.parquet` 是 E1 给 E6 的正式 raw-score 交接文件。每份文件各有 5,086 行，包含 D1 Validation、D1/D2/D3 Test 和 D4；每个模型、样本只有一行，没有预先应用阈值。E6 可以直接使用 D1 Validation 行独立完成阈值选择，并用 joblib 模型复算核验：

- `tF1`：在 D1 Validation 上最大化 Macro F1；并列时取更高阈值。
- `t5`：在 D1 Validation 经验 FPR ≤ 5% 的候选中最大化 Attack Recall；并列时取更高阈值。
- 预测规则：`pred = 1` 当且仅当 `score_attack >= threshold`。

冻结阈值后，再评测 D1、D2、D3 Test。D4 全部是正常类，只报告 FP/N、FPR、分组结果和二项置信区间，不报告 F1 或 AUROC。

`results/e1/predictions` 保留的是 E1 原始实验的可追溯 Test prediction，同一个测试样本按照 `default_0.5`、`tF1`、`t5` 保存三行。E6 的正式输入应优先使用 `results/e1/handoff/*_scores.parquet`；若核对旧 Test prediction，可先选择：

```python
predictions = pd.read_parquet(
    "results/e1/predictions/combined_predictions.parquet"
)
unique_test_scores = predictions[
    predictions["threshold_policy"].eq("default_0.5")
].copy()
```

已有的测试预测可以作为核对材料，但 E6 的正式阈值、D4 结果和置信区间应由 E6 自己生成。

### 错误分析、结果整合和应用成员

- 错误案例负责人：使用 `results/e1/predictions/*.parquet` 与对应的 D1/D2/D3 processed Test，通过 `sample_id` 恢复文本。
- 跨来源主表负责人：收集 E1–E4 相同 schema 的预测文件，分别报告 D1、D2、D3，不能把三个来源合并成一个分数。
- Streamlit 应用负责人：使用 `models/e1/*.joblib`、`configs/e1.json` 和 `results/e1/selected_models.csv`；界面应显示模型、分数、冻结阈值、预测和必要说明，不使用“绝对安全”措辞。

## 8. 各实验完成后交给 E6 的材料

E6 需要的是**冻结模型的原始攻击分数**，不是各成员自行选择阈值后的最终标签。E2–E4 完成模型选择后，应为最终选定的 checkpoint 生成 D1 Validation、D1 Test、D2 Test 和 D3 Test 分数；E6 再统一选择 `tF1` 和 `t5`。

每个实验建议建立：

```text
results/<experiment>/handoff/
├── selected_config.json
├── tuning.csv
├── <model_id>_scores.parquet
├── run_metadata.json
└── checkpoint/ 或 checkpoint_path.txt
```

必须交给 E6 的内容如下：

| 提供者 | 必须交付 | 说明 |
|---|---|---|
| E2 | 选定的 BiLSTM checkpoint、`selected_config.json`、`tuning.csv`、选定模型的 `scores.parquet`、`run_metadata.json` | 分数文件包含 D1 Validation 和 D1/D2/D3 Test；同时报告初始化方式、词表/OOV、seed |
| E3 | 选定的 DistilBERT checkpoint 与 tokenizer revision、`selected_config.json`、`tuning.csv`、`scores.parquet`、`run_metadata.json` | 记录 learning rate、epoch、截断率、模型 revision 和 seed |
| E4 | 选定的 RoBERTa checkpoint 与 tokenizer revision、`selected_config.json`、`tuning.csv`、`scores.parquet`、`run_metadata.json` | 同时记录 latency、P95、显存、模型大小和截断率 |
| E5 | Clean 与 Augmented 的配置、checkpoint、原始/扰动配对分数、D1/D2/D3/D4 分数、`run_metadata.json` | `parent_id` 和 `transform_id` 必须保留；若 E6 主表只分析六条主管线，E5 结果仍用于独立增强与 D4 误报表 |

E2–E4 交给 E6 的 `<model_id>_scores.parquet` 应为每个模型、每个样本一行，至少包含：

```text
dataset, split, sample_id, gold, score_attack,
model_id, checkpoint_rev, seed, transform_id, parent_id
```

要求：

- D1 Validation 必须包含 1,985 行，这是 E6 选择阈值的唯一数据。
- 最终选定模型的 D1 Test、D2 Test、D3 Test 分别应有 2,101、116、545 行。
- `score_attack` 越大必须表示越倾向攻击类；分数必须为有限数值。
- `(model_id, dataset, split, sample_id, transform_id)` 不得重复。
- `sample_id` 必须与 processed 数据完全一致。
- 交付给 E6 前不得根据任何 Test 或 D4 结果更换 checkpoint。
- 如果不传 checkpoint 文件，必须提供组员可访问的路径、文件哈希和精确加载说明。

E1 已提供 `results/e1/handoff/word_scores.parquet`、`char_scores.parquet` 和 `combined_scores.parquet`。每份文件包含 1,985 条 D1 Validation、2,101 条 D1 Test、116 条 D2 Test、545 条 D3 Test 和 339 条 D4 raw scores。E6 应使用其中的 D1 Validation 行选择阈值，并可用 `models/e1/*.joblib` 独立复算。

E6 收齐各模型的 raw scores 后，应向全组返回：

```text
thresholds.csv
metrics.csv
predictions.parquet
notinject_false_alarms.csv
run_metadata.json
```

其中 `thresholds.csv` 记录每个模型的 `tF1` 和 `t5`、D1 Validation 选择指标和并列规则；`metrics.csv` 分别报告 D1、D2、D3 的 Macro F1、Attack Recall、FPR 与计数；`notinject_false_alarms.csv` 报告 D4 的 FP/N、FPR、分组结果和置信区间。

## 9. E6 最终预测 schema

E6 应在 raw scores 上添加阈值政策和最终预测，至少保留：

```text
dataset, split, sample_id, gold, score_attack, threshold_policy,
threshold, pred, model_id, checkpoint_rev, seed, transform_id, parent_id
```

E5 汇总跨来源主表和其他成员进行错误分析时，应使用 E6 返回的统一阈值版本，避免每个成员各自选择阈值。

## 10. 团队成员之间的交接关系

下面的“提供者”负责生成文件，“接收者”负责检查 schema、行数、模型 ID 和 checkpoint 信息；接收不代表可以用测试集重新调参。

| 提供者 | 接收者 | 必须交付 | 用途 |
|---|---|---|---|
| M1 / E1 与数据负责人 | M2、M3、M4 | D1 Train、D1 Validation、manifest、integrity report、数据字段说明 | 使用同一训练/验证 ID 开展 E2–E4 |
| M1 | M2、M3、M4 | 配置冻结后解锁 D1/D2/D3 Test | 统一跨来源测试 |
| M1 | M5 | `models/e1/combined.joblib`、`configs/e1.json`、`selected_models.csv`、D1 Train/Validation/Test、D2/D3/D4 | 建立 Clean 与 Augmented 对照 |
| M1 | M6 | 三个 E1 joblib、`results/e1/handoff/*_scores.parquet`、`selected_models.csv`、E1 配置 | 核验 E1 raw scores 并统一选择阈值 |
| M2 / E2 | M6 | 选定 BiLSTM checkpoint、配置、tuning、D1 Validation 与 D1/D2/D3 raw scores、metadata | 计算 E2 的 `tF1`、`t5` 和跨来源结果 |
| M3 / E3 | M6 | 选定 DistilBERT checkpoint/tokenizer、配置、tuning、raw scores、metadata | 计算 E3 阈值与结果 |
| M4 / E4 | M6 | 选定 RoBERTa checkpoint/tokenizer、配置、tuning、raw scores、metadata | 计算 E4 阈值与结果 |
| M5 / E5 | M6 | Clean/Augmented raw scores、D4 scores、augmentation manifest、配对 ID | 如团队纳入增强模型，使用相同阈值逻辑；支持 D4 误报对照 |
| M6 / E6 | M5 与结果整合成员 | `thresholds.csv`、`metrics.csv`、统一 `predictions.parquet`、D4 误报表 | 生成六条主管线 × 三来源主表及 E5 补充表 |
| M6 | M2、M4 | 统一阈值后的预测、FP/FN 候选 ID、模型和阈值字段 | 六案例错误分析与局部解释 |
| M6 | M3 / 应用负责人 | 每个演示模型的冻结阈值、`model_id`、checkpoint 信息 | Streamlit 显示正确阈值、分数和预测 |
| M2、M3、M4 | M5 / 主表整合负责人 | 各自效率数据、selected model metadata 和 E6 返回的统一指标 | 效果与成本主表 |
| 所有成员 | 报告整合与视频负责人 | 200–400 字 findings/limitations、一个实测图、复现命令、配置与贡献证据 | 形成最终小组报告、视频和贡献声明 |

推荐交接顺序：

```text
M1 冻结数据和 E1
  → M2/M3/M4 训练并选 checkpoint
  → M2/M3/M4 向 M6 交 raw scores
  → M5 完成 Clean/Augmented 并交配对结果
  → M6 统一选阈值、评测和核验 ID
  → M6 把统一结果交给 M2/M3/M4/M5
  → M5 汇总主表，M2/M4 完成案例与解释，M3 完成应用
  → 全员提交文字、图和贡献证据给报告整合负责人
```

## 11. 每位成员最终必须输出

### 所有成员的共同输出

每位成员都必须提供：

1. 可运行的实验脚本和一条明确的执行命令。
2. 完整配置文件，包括 seed、依赖版本和数据/model revision。
3. tuning 或受控对照的完整记录，不能只保留最好结果。
4. 选定模型/checkpoint 及 SHA-256 或可访问路径。
5. D1 Validation raw scores，以及配置冻结后的测试 raw scores。
6. 一张来自真实输出的结果图。
7. 200–400 英文词的 Findings and Limitations。
8. `run_metadata.json` 和贡献证据，例如 commit、CSV、checkpoint 或图。

### M1 / E1 与数据负责人

最终输出：

- `data/processed/d1_train.parquet`
- `data/processed/d1_validation.parquet`
- `data/processed/d1_test.parquet`
- `data/processed/d2_test.parquet`
- `data/processed/d3_test.parquet`
- `data/processed/d4_test.parquet`
- `data/audit/manifest.json`
- `data/audit/integrity_report.json`
- `configs/e1.json`
- 三个 E1 joblib 模型
- `results/e1/evidence/tuning.csv`、`selected_models.csv`、三条管线预测和 `results/e1/handoff/*_scores.parquet`
- `results/e1/evidence/` 中的特征维度、全局系数、局部贡献、结果图、运行摘要、复算记录和 findings/limitations
- `results/e1/reference_only/` 中的旧阈值/指标仅作追溯；正式版本由 E6 返回

验收重点：三种表示使用同一 C 网格；C 只由 D1 Validation、固定阈值 0.5 选择；D2/D3/D4 不参与模型选择。

### M2 / E2 BiLSTM

最终输出：

- Random 与 GloVe 两种初始化的完整配置和训练记录
- 训练/验证曲线
- 词表大小、GloVe coverage 和 OOV rate
- 选定 checkpoint 与 selection 记录
- D1 Validation 和 D1/D2/D3 raw scores
- batch-1 latency、模型大小和运行环境
- 六案例错误分析的案例表，包含 `sample_id`、gold、prediction、excerpt 和解释
- E2 结果图及 findings/limitations

验收重点：两种条件只改变 embedding 初始化；词表来自 D1 Train；不能根据 Test 选择初始化或 epoch。

### M3 / E3 DistilBERT 与应用

最终输出：

- 两个 learning rate 条件的配置、训练/验证曲线和 tuning 表
- 选定 checkpoint、model revision、tokenizer revision、epoch 和 seed
- 截断率、batch-1 latency、模型大小和运行环境
- D1 Validation 和 D1/D2/D3 raw scores
- 可运行的 Streamlit 应用、启动命令和模型缓存说明
- 应用至少支持两个冻结模型，并显示 score、threshold、prediction 和 truncation notice
- E3 结果图及 findings/limitations

验收重点：checkpoint 只由 D1 Validation 在阈值 0.5 下选择；测试后不得更换学习率或 checkpoint。

### M4 / E4 RoBERTa 与局部解释

最终输出：

- 两个 learning rate 条件的配置、训练/验证曲线和 tuning 表
- 选定 checkpoint、model/tokenizer revision、epoch 和 seed
- 截断率、模型大小、median/P95 latency 和 peak VRAM
- D1 Validation 和 D1/D2/D3 raw scores
- 固定案例的 token/span occlusion 结果和图
- 对遮蔽结果的限制说明，不能把它描述为因果解释
- E4 结果图及 findings/limitations

验收重点：E3 与 E4 使用相同的学习率候选数和 checkpoint 选择规则；成本计时范围必须披露。

### M5 / E5 Augmentation 与主表整合

最终输出：

- Clean 与 Augmented 的完整配置和模型
- augmentation manifest，包含 `sample_id`、`parent_id`、`transform_id`、seed、变换位置和有效性
- typo 人工 QA 记录、无效数量和最终配对分母
- duplicate-equivalence check
- 原文/typo 配对 raw scores 和 predictions
- D1/D2/D3/D4 的 Clean/Augmented raw scores 与 Recall/FPR
- 有效表示变化率
- E1–E4 六条主管线 × D1/D2/D3 的跨来源主表
- E5 结果图及 findings/limitations

验收重点：固定 E1 Combined vectorizer、词表、IDF 和 `C=1`；只改变训练增强；训练和评测使用不同 augmentation seeds。

### M6 / E6 阈值、误报与统一评测

最终输出：

- 统一 scorer 和 ID/行数核验脚本
- 每个主管线的 D1 Validation raw score 验收记录
- `thresholds.csv`：每个模型的 `tF1`、`t5`、选择指标和并列规则
- `metrics.csv`：D1/D2/D3 的 Macro F1、Attack Recall、FPR、TP/FP/TN/FN
- 统一 `predictions.parquet`
- `notinject_false_alarms.csv`：D4 总体、trigger-count 和 category 切片的 FP/N、FPR 与二项置信区间
- `tF1` 对 `t5` 的 Recall–FPR 表或图
- `run_metadata.json`、完整环境和复现命令
- E6 结果图及 findings/limitations

验收重点：阈值只能用 D1 Validation 选择；D2/D3/D4 仅评测；D4 不报告 F1、AUROC 或 AUPRC。

## 12. 推荐的统一目录结构

每位成员应把自己的新增文件放在独立目录，不覆盖其他成员结果：

```text
configs/e2.json ... configs/e6.json
experiments/e2/ ... experiments/e6/
models/e2/ ... models/e5/
results/e2/
├── handoff/
├── tuning.csv
├── selected_config.json
├── run_metadata.json
├── figures/
└── findings_and_limitations.md
```

E6 的统一输出建议放在：

```text
results/e6/
├── thresholds.csv
├── metrics.csv
├── predictions.parquet
├── notinject_false_alarms.csv
├── verification.json
├── run_metadata.json
└── figures/
```

## 13. 每个交接包的验收清单

提供者和接收者应共同确认：

- [ ] 文件能在项目根目录使用相对路径读取。
- [ ] 配置、checkpoint、tokenizer/model revision 和 seed 已记录。
- [ ] D1 Validation raw scores 行数为 1,985，`sample_id` 唯一且与 processed 数据一致。
- [ ] D1/D2/D3 Test 行数分别为 2,101、116、545。
- [ ] `score_attack` 为有限数值，且数值越大越倾向攻击。
- [ ] 测试分数是在 checkpoint 冻结后一次性生成。
- [ ] prediction schema 与第 8、9 节一致。
- [ ] 没有使用 D2/D3/D4 选择模型或阈值。
- [ ] latency 说明硬件、batch size、warm-up、计时次数和是否包含 tokenization。
- [ ] findings、图、CSV 和正文引用的是同一 checkpoint。
- [ ] 未包含 `.DS_Store`、`__pycache__`、`.venv`、密钥或临时文件。

## 14. 当前交接包状态

- 六个 processed 数据文件已通过 manifest 哈希核验。
- `integrity_report.json` 状态为 `passed`。
- 三个 E1 joblib 模型均可加载，并可对 D1 Validation 和 D4 生成分数。
- E1 的 tuning、解释证据、图、运行摘要和复算记录已归档在 `results/e1/evidence`。
- E1 的 `selected_models.csv`、三个可追溯 Test prediction 和三份正式 raw-score handoff 文件位于 `results/e1`；完整阈值与统一指标由 E6 生成。
- `results/e1/reference_only` 仅保留历史 E1 阈值/指标用于核对，不作为最终报告数据源。
- 本目录不包含单独的 E1 报告；报告内容由组员整合进最终小组报告。

请勿提交 `.DS_Store`、`__pycache__`、`.venv` 或以 `~$` 开头的临时文件。

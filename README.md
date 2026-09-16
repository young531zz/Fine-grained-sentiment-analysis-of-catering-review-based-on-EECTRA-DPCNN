# 基于 ELECTRA-DPCNN 的餐饮评论细粒度情感分析

本仓库整理自百度飞桨 AI Studio 实验，对应论文：

> Cheng, Y., & Wang, L. *Fine Grained Sentiment Analysis of Catering Reviews Based on ELECTRA-DPCNN Model*.  
> 全文见 [`paper/Fine-grained-sentiment-analysis-ELECTRA-DPCNN.pdf`](paper/Fine-grained-sentiment-analysis-ELECTRA-DPCNN.pdf)

任务是**方面级情感分析（Aspect-Based Sentiment Analysis, ABSA）**：给定一条餐饮评论和一个预定义方面（如 `Food#Taste`、`Service#Hospitality`），判断该评论在**这一方面**上的情感极性。

| 极性 | 原始标签 | 训练时映射 |
|------|----------|------------|
| 负面 | `-1` | `0` |
| 中性 | `0` | `1` |
| 正面 | `1` | `2` |

论文在 ASAP-ASPECT 测试集上报告的结果为 **Accuracy 86.64% / F1 82.07%**（本仓库未重新训练，数字来自论文）。

---

## 问题背景：为什么不用整句情感分类？

餐饮评论经常在同一句话里同时夸和贬，例如「味道很好，但是排队太久」。整句（coarse-grained）分类只能给出一个极性，无法告诉商家「菜品好、服务差」。

ABSA 把问题拆成：

1. **方面（aspect）**：评价对象属于哪一类属性  
2. **极性（polarity）**：对该属性是正面、中性还是负面

本数据集的方面已经标注好（`cate` 列），因此本项目做的是 **aspect-given 的极性分类**，而不是开放式方面抽取。

同一条评论会按它涉及的多个方面展开成多行。例如一条关于空间、口味、性价比的评论，在训练集中会变成三行，`text_a` 相同、`cate` 和 `label` 不同。

输入拼接方式与 BERT 句对分类相同：

```text
[CLS] 评论文本 [SEP] 方面类别 [SEP]
```

`token_type_ids` 把评论与方面区分开，模型必须结合「这段话」和「问的是哪个方面」才能给出正确极性。

---

## 方法思路

餐饮评论有两个难点：

- **噪声多**：口语、错别字、活动标签（如「大众点评霸王餐」）、表情和无关铺垫  
- **文本长、线索远**：平均约 300+ 字，情感词可能离方面词很远；`max_seq_len=128` 会截断，因此后级网络需要在有限序列里做多尺度抽象

整体结构是 **预训练编码器 + 深度金字塔卷积**：

```text
评论文本 + 方面类别
        │
        ▼
 ELECTRA-small（语义编码）
        │  hidden: [batch, seq_len, 256]
        ▼
     维度转置 → [batch, 256, seq_len]
        │
        ▼
 改进 DPCNN
   Region Embedding（k=3 卷积）
        → 窄卷积（k=1）
        → 残差卷积块
        → MaxPool 下采样（金字塔）
        → 池化后再接窄卷积
        → 自注意力
        │
        ▼
 自适应平均池化 → Dropout → Linear(256 → 3)
        │
        ▼
   交叉熵三分类
```

### 为什么用 ELECTRA 而不是 BERT？

BERT 的预训练是 MLM：随机遮住约 15% 的 token，只在这些位置算损失。ELECTRA 改成 **Replaced Token Detection (RTD)**：

1. 小生成器先做 MLM，把部分 token 替换成它的预测  
2. 判别器对**每一个** token 做二分类：是原文还是被替换过的  

同样的计算量下，判别器能从整句所有位置学习，样本效率更高。论文在同一数据上比较了若干中文预训练模型，ELECTRA 单独微调准确率为 **82.30%**，高于 BERT / RoBERTa / ALBERT / ERNIE，因此用它做编码器。

本实现使用 PaddleNLP 的 `electra-small`（隐层 256 维），与代码中 `hidden_size = 256` 对齐。

### 为什么在 ELECTRA 后面接 DPCNN？

ELECTRA 给出的是 token 级语义。餐饮评论还需要：

- **局部 n-gram**（「味道不错」「排队太久」）  
- **更长距离的转折**（先扬后抑）  
- **压低无关噪声**  

DPCNN（Deep Pyramid CNN, Johnson & Zhang, 2017）用卷积 + 步长为 2 的池化不断加深感受野，计算量增长慢于直接堆很深的 CNN，适合偏长的文本分类。

论文相对经典 DPCNN 做了三处改动（代码中均保留）：

| 模块 | 作用 |
|------|------|
| Region Embedding 后的 **k=1 窄卷积** | 在不扩大窗口的情况下混合通道，细化局部特征 |
| 下采样后的 **k=1 窄卷积** | 序列变短后再次整合通道，减轻池化带来的信息损失 |
| 金字塔末端的 **自注意力** | 对压缩后的序列做 `Q/K/V` 加权，突出与当前方面相关的片段 |

单独用 DPCNN 约 73.56%，加上自注意力约 74.10%；接到 ELECTRA 之后，论文报告测试准确率升至 **86.64%**。对比实验见下表（数字来自论文，非本仓库复现）。

| 模型 | Accuracy (%) | F1 (%) |
|------|-------------:|--------:|
| BERT | 81.13 | — |
| ELECTRA | 82.30 | — |
| DPCNN | 73.56 | — |
| Self-Attention DPCNN | 74.10 | — |
| ERNIE-BiLSTM | 83.82 | 79.05 |
| ERNIE-DPCNN | 84.75 | 82.83 |
| ELECTRA-TextCNN | 85.21 | 80.35 |
| **ELECTRA-DPCNN** | **86.64** | **82.07** |

### 训练策略

- 损失：交叉熵；标签先由 `{-1,0,1}` 加 1 映射到 `{0,1,2}`，预测时再减 1 还原  
- 优化器：AdamW，学习率 `2e-5`，L2 / weight decay `1e-4`  
- 按验证准确率存 `outputs/best_model.pdparams`  
- 连续 15 个 epoch 无提升则早停（最多 50 epoch）

默认使用官方 `dev.tsv` 做验证。原 AI Studio 脚本是从训练集 8:2 随机切分、未使用官方验证集；若要对齐旧实验，把笔记本里的 `config.use_official_dev` 设为 `False`。

---

## 数据集：ASAP-ASPECT

来源为美团点评构建、百度千言（千言数据集）发布的中文细粒度情感数据 **ASAP-ASPECT**，领域是餐饮评论。

> 论文把 train / dev 的样本量写反了。下表以实际文件为准。

| 划分 | 文件 | 样本数 | 独立评论数 | 表头 |
|------|------|--------:|----------:|------|
| 训练 | `trian.tsv`（原始文件名拼写如此） | 213,371 | 36,802 | `text_a`, `cate`, `label` |
| 验证 | `dev.tsv` | 29,101 | 4,938 | `qid`, `text_a`, `cate`, `label` |
| 测试 | `test.tsv` | 28,362 | 4,931 | `qid`, `text_a`, `cate`（无标签） |

训练文件名是 `trian.tsv` 而不是 `train.tsv`。笔记本会自动识别两种文件名。

文本长度大约：最短 50+ 字，平均约 335 字，最长约 1000 字。`max_seq_len=128` 会对长评截断，这是原实验设置。

### 18 个方面类别

方面采用 `大类#细类` 命名，覆盖口味、环境、价格、服务、位置：

| 大类 | 细类 |
|------|------|
| Food | Taste, Portion, Appearance, Recommend |
| Service | Hospitality, Timely, Queue, Parking |
| Ambience | Space, Noise, Decoration, Sanitary |
| Price | Level, Cost_effective, Discount |
| Location | Transportation, Downtown, Easy_to_find |

训练集方面分布不均匀：`Food#Taste`（34,872）最多，`Service#Parking`（2,476）最少。

### 标签分布（训练集）

| 标签 | 含义 | 条数 | 占比 |
|------|------|-----:|-----:|
| `1` | 正面 | 133,721 | 62.7% |
| `0` | 中性 | 52,225 | 24.5% |
| `-1` | 负面 | 27,425 | 12.8% |

类别不平衡，准确率会被正面类抬高；论文同时报告了 F1。

### 样本形态

训练集一行三列：

```text
text_a                                          cate                 label
第一次去…店面宽敞明亮，地方比较大…              Ambience#Space       1
第一次去…一开始先品尝…味道不错…                 Food#Taste           1
```

验证集多一列 `qid`；测试集有 `qid` / `text_a` / `cate`，没有 `label`。

### 如何放置数据

仓库根目录包含 `ASAP_ASPECT.zip`（约 26MB，低于 GitHub 100MB 单文件限制）。解压后的 `trian.tsv` 约 211MB，因此**只提交压缩包，不提交解压后的 TSV**。

克隆后可直接使用 zip；笔记本会自动解压到 `data/ASAP_ASPECT/`：

```text
data/ASAP_ASPECT/
├── trian.tsv    # 或 train.tsv
├── dev.tsv
└── test.tsv
```

数据查找顺序：

1. 若 `data/ASAP_ASPECT/` 中还没有划分文件，且根目录存在 `ASAP_ASPECT.zip`，则自动解压到 `data/`  
2. 否则依次尝试 `data/ASAP_ASPECT/` 与 AI Studio 挂载目录 `/home/aistudio/data/data347891`

---

## 代码结构

```text
.
├── electra_dpcnn_absa.ipynb   # 整理后的主实验笔记本（请使用这个）
├── 9342991.py                 # AI Studio 原始导出，仅作对照
├── ASAP_ASPECT.zip            # 数据集压缩包
├── requirements.txt
├── paper/                     # 论文 PDF
├── data/ASAP_ASPECT/          # 解压后的数据（gitignore）
└── outputs/                   # 权重、曲线、预测结果（gitignore）
```

主文件是 `electra_dpcnn_absa.ipynb`，按实验流程分成 7 段：

| 段落 | 内容 |
|------|------|
| 1. 环境与数据准备 | 定位 / 解压数据，导入 Paddle、PaddleNLP |
| 2. 配置 | `Config` 超参与路径 |
| 3. 数据集 | `AspectDataset`：读 TSV、标签映射、ELECTRA tokenize |
| 4. 模型 | `DPCNN`、`ElectraDPCNN` |
| 5. 训练 | 官方验证集或随机切分、早停、画曲线 |
| 6. 预测 | 加载最优权重，写出 `outputs/ASAP_ASPECT.tsv` |
| 7. 运行 | `DO_TRAIN` / `DO_PREDICT` 开关 |

关键类与函数：

- `AspectDataset`：按表头读取；训练/验证需要 `text_a, cate, label`，测试需要 `qid, text_a, cate`  
- `DPCNN`：Region Embedding → 窄卷积 → 残差块 → 下采样 → 自注意力  
- `ElectraDPCNN`：ELECTRA 最后一层隐状态转置后送入 DPCNN，再池化分类  
- `train()`：保存验证集上准确率最高的权重  
- `predict()`：类别索引减 1，还原为 `{-1, 0, 1}`

相对原始 `9342991.py` 的主要整理：

- 去掉 AI Studio 模板、`get_ipython()` 和重复定义的模型/训练函数  
- 按表头读数据（不再把第一行当样本）  
- 测试集字段使用 `qid`  
- 默认用官方 `dev.tsv`  
- 设备统一 `paddle.set_device`，结果写入 `outputs/`

---

## 环境与运行

论文实验环境：百度 AI Studio，PaddlePaddle 2.4.0，Python 3.7，Tesla V100 16GB。

本地或新平台：

```bash
pip install -r requirements.txt
```

GPU 可将 `requirements.txt` 中的 `paddlepaddle` 换成 `paddlepaddle-gpu`。还需能下载 PaddleNLP 的 `electra-small` 预训练权重。

1. 克隆仓库（根目录已有 `ASAP_ASPECT.zip`）；首次运行笔记本时会自动解压  
2. 用 Jupyter / VS Code / AI Studio 打开 `electra_dpcnn_absa.ipynb`  
3. 自上而下运行；第 7 段默认 `DO_TRAIN = True`、`DO_PREDICT = True`  

完整 50 epoch 在 V100 上耗时较长。通路检查可先把 `config.num_epochs` 调小。只做推理时：确认 `outputs/best_model.pdparams` 存在，设 `DO_TRAIN = False`。

在 AI Studio 上，若数据已挂载到 `/home/aistudio/data/data347891`，无需改路径。

输出文件：

| 路径 | 说明 |
|------|------|
| `outputs/best_model.pdparams` | 验证集最优权重 |
| `outputs/training_loss_curve.png` | 训练损失 |
| `outputs/validation_accuracy_curve.png` | 验证准确率 |
| `outputs/combined_metrics.png` | 以上两图合并 |
| `outputs/ASAP_ASPECT.tsv` | 测试集预测，`qid` + `prediction` |

---

## 超参数

与论文第四节一致：

| 参数 | 值 | 说明 |
|------|----|------|
| `pretrained_model` | `electra-small` | PaddleNLP ELECTRA |
| `hidden_size` | 256 | 与 electra-small 隐层一致 |
| `max_seq_len` | 128 | 超长评论会被截断 |
| `batch_size` | 32 | |
| `num_epochs` | 50 | 上限，可能早停 |
| `learning_rate` | `2e-5` | AdamW |
| `l2_lambda` | `1e-4` | weight decay |
| `num_classes` | 3 | 负 / 中 / 正 |
| `conv_filters` | 256 | DPCNN 通道数 |
| `kernel_size` | 3 | Region / 残差卷积核 |
| `dropout_prob` | 0.1 | 分类头前 |
| `patience` | 15 | 早停 |
| `use_official_dev` | `True` | `False` 则训练集 8:2 随机切分 |
| `seed` | 42 | |

---

## 使用与复现注意

- **标签映射**：训练 `label + 1`，预测 `argmax - 1`，提交文件中的值应回到 `{-1, 0, 1}`。  
- **类别不平衡**：正面约占 63%，只看 Accuracy 会偏乐观。  
- **长文本截断**：平均 335 字、上限 128 token，后半段评论可能进不了模型。  
- **验证协议**：本笔记本默认官方 `dev.tsv`；论文曲线对应的是原脚本的随机切分，二者数值不可直接对比。  
- **论文表 II/III/IV** 是作者完整实验；本仓库只实现最终的 ELECTRA-DPCNN，不包含全部基线。  
- `9342991.py` 保留原始导出（含重复代码与 AI Studio 路径），复现与二次开发请用笔记本。

---

## 引用

若使用本方法或数据集，请同时参考原论文与 ASAP-ASPECT / 百度千言数据发布信息。

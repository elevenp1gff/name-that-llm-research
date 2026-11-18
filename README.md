# Name That LLM — 18 类 LLM 文本归因

Kaggle 竞赛 `name-that-llm` 的完整实验记录、可复现代码与结论。
指标为 **balanced accuracy**（宏平均召回），代码赛（提交 notebook，离线重跑约 4,000 条隐藏文本）。

> 本仓库只含代码、文档与聚合后的实验数字。**不含**竞赛数据、训练权重、预测文件，也不含任何凭证或主机信息。

---

## 结果

| 提交 | 改动 | public LB |
|---|---|---:|
| 复现公开 best notebook | — | **0.668** |
| + 自训 LoRA 单成员（w=0.8/0.4） | 见 §4 | 0.670 |
| + 自训 LoRA 5 折 bag（w=1.2） | | 0.664 |
| SVC 换 204k 并集池 | | 0.667 |
| **hard assignment 转导** | 见 §5 | **0.525** |
| soft centring 阈值校准 | `adapt_thr 0.15 → 0.05` | 待评分 |

赛事在此期间榜首从 0.747 升到 0.768。**本项目最好成绩 0.670，未进入前十。** 原因与证据见 §7。

---

## 1. 问题设定

训练集 2,699 条 / 150 个 prompt × 18 个模型（其中 1 个 prompt 缺 1 条）。
隐藏测试约 4,000 条，**prompt 与训练集不重叠**，且每个 prompt 的答案数**不相等**。

18 个标签（厂商 / 模型）：`claude-*`(5) `gemini-*`(4) `gpt-*`(5) `glm-*`(2) `kimi-k3`(1) `qwen3.8-max`(1)。

因为指标是宏平均召回，**每个类等权**，所以稀有类/难类的召回直接决定分数（可用下述诊断定位）。

---

## 2. 验证协议（本项目的地基）

`corpus/cv.py` 提供唯一真源，所有实验必须经过它：

- `prompt_folds(official, n_splits=5, seed)` — 按 `prompt_id` 分组的 GroupKFold，贪心按类负载均衡。校验：折间 prompt 交叉 = 0，每折 18/18 类、每类 30 条。
- `sim_subset(idx, official, lo, hi, seed)` — **模拟隐藏测试的残缺面板**：每个 prompt 只保留 1–2 条。这是本赛最关键的设计，因为隐藏 test 不是完整 18 面板（§5 有实证）。
- `evaluate(y_true, y_pred)` — 返回 `bacc` 与逐类召回。
- `board_like_score(...)` — 多次重采样 1–2 条/prompt 的宏召回均值，比 full-OOF 更贴近榜单。

自检：完美预测 1.0000；随机打乱 0.0530 ≈ 1/18。

---

## 3. 数据：公开 pool 的并集

四个公开 pool 合并，**精确文本去重**（collapse whitespace + lowercase，按来源优先级 keep first），再过滤 40–250 词：

```
raw concat          512,038
after dedupe        373,200
after length filter 269,120
capped per-label    165,208   (cap=9,486, max/min=1.60)
```

来源优先级：`crtproduction open pool` → `olegzholobov response corpus`（宽表需 melt）→ `prvsiyan clean passages` → `nuridudak 130k`。

**泄漏检查（已验）**：与 `train.csv` 精确文本重叠 **0**；与 3 条可见测试行文本重叠 **0**、prompt 重叠 **0**；与 150 条重建 train prompt 的 TF-IDF 余弦最大 0.678，>0.7 **0 对**。

**未采用**：`gpt-5.6-luna` 的作者自述标签文件（74,790 行，标签非独立）。

### 关键实测：数据量是最大的杠杆

同一个 4-view n-gram SVC，只改每类封顶数：

| 每类上限 | pool 行数 | full OOF | board-like |
|---|---:|---:|---:|
| — (仅 official) | 2,699 | 0.2615 | 0.2690 |
| 3,000 | 54,000 | 0.3993 | 0.3946 |
| 9,000 | 157,432 | 0.4515 | 0.4486 |
| 全量 | 269,120 | **0.4793** | **0.4760** |

单调上升，**未饱和**。

### 反直觉：LoRA 对语料规模几乎不敏感

同一 5 折协议，只换 stage-1 语料：

| stage-1 语料 | 5 折真 OOF 均值 |
|---|---:|
| 165,208 行 | 0.5875 |
| 269,120 行 | 0.5908 |

**+0.003**，而 n-gram 是 +0.028。语料规模对 LoRA 的边际收益约为 n-gram 的 **1/9**。
（这否掉了我自己"扩语料重训 LoRA"的计划——那是把 n-gram 的曲线外推到另一个模型类。）

### kb_proxy 数据是否污染

公开 pool 有 72,030 行经 Benchmarks 代理生成（排版与真实 API 不同）。实测：

| 训练池 | 行数 | full OOF |
|---|---:|---:|
| 全部 | 269,120 | **0.4793** |
| 剔除 kb_proxy | 197,090 | 0.4649 |
| 仅 kb_proxy | 72,030 | 0.4223 |

**剔除反而变差** → 这些行不是污染，是有效数据（量大胜于纯）。

---

## 4. 训练：两阶段截断 LoRA 分类器

`train/lora24.py` + `train_lora.py`（Ascend NPU 实现，手写 LoRA 以绕开环境限制）。

- **截断骨干**：Qwen3-14B-Base 保留前 27/40 层，Qwen3-8B-Base 保留 24/36 层。
  依据：风格信息在早/中层，末层编码主题，而测试 prompt 是新的。
- **LoRA** r=16 / α=32 / dropout 0.05，打在 q/k/v/o + gate/up/down。
- **池化**：8 个深度（`[2,4,6,8,12,16,20,24]`）各做 masked mean（**排除 token 0**，attention sink），逐层 LayerNorm 后拼接。
- **双头**：`plain` + `centred`（对同 prompt 组减去组均值），loss = CE(plain) + CE(centred)，类加权。
- **两阶段**：stage1 用大 pool（1 epoch, LoRA lr 1e-4 / head 1e-3）；stage2 用 official（2 epochs, lr 5e-5 / 5e-4）。bf16 + gradient checkpointing。
- 实测：1.47 s/step @ bs8 / 384 tok，峰值 HBM 21.3 GB；stage1 18,357 步 ≈ 7.5h，stage2 600 步 ≈ 15min。

**真 OOF（逐折重训，每折只见 4/5 official）**：

| fold | 0 | 1 | 2 | 3 | 4 | 均值 |
|---|---:|---:|---:|---:|---:|---:|
| bacc | 0.6019 | 0.5741 | 0.5963 | 0.6000 | 0.5652 | **0.5875** |

折间极差 0.037，印证"不要读小于 ~0.01 的差距"。

> 注意：正式提交用的两个全量检查点 stage2 见过全部 official，因此**无法自证 OOF**，只能靠实盘；上面的 0.5875 来自独立的逐折重训实验。

---

## 5. 转导（transductive）：本项目最重要的发现

隐藏 test 无 `prompt_id`，但可以从文本**恢复** prompt 分组。
实测恢复质量（内容词 TF-IDF + 平均链接，相似度阈值 0.05）：**pair precision 1.000、recall 1.000**。

拿到分组后有两种利用方式，性能天差地别：

| 每 prompt 答案数 | 簇纯度 | plain | **soft centring** | hard assignment |
|---|---:|---:|---:|---:|
| 18 | 1.000 | 0.3063 | **0.3745 (+0.068)** | 0.3990 |
| 12 | 1.000 | 0.3121 | **0.3720 (+0.060)** | 0.3521 |
| 8 | 0.666 | 0.3185 | **0.3484 (+0.030)** | 0.3170（负） |
| 5 | 0.515 | 0.2873 | **0.2923 (+0.005)** | 0.2752（负） |
| 3 | 0.420 | 0.2868 | 0.2781（−0.009） | 0.2542（负） |

- **soft centring**（只用簇均值中心化，不施加互斥约束）在 k≥5 时**全部为正**。
- **hard assignment**（Hungarian，强制 18 类互斥）在 k≤8 时转负。

### 实盘检验

我在公开 0.668 管线上把 hard 路径打开（修掉一处 `--post`/`--perpassage` 互斥导致的死代码），提交得 **0.525（−0.143）**。

**这条负结果证明了隐藏 test 是残缺面板**，并解释了为什么论坛上多个团队报告 assignment 掉 0.009–0.017。
它同时确立了本项目最重要的一条纪律：**当离线 CV 与多个独立实盘冲突时，信实盘。**

### 由此得到的可提交改动

`adapt_thr`（恢复分组的相似度阈值）在公开管线里默认 0.15，**从未被校准**。扫描结果：

| thr | k18 | k12 | k8 | k5 | k3 | 全非负 |
|---|---:|---:|---:|---:|---:|---|
| 0.02 | +0.068 | +0.060 | +0.030 | +0.005 | −0.009 | ✗ |
| **0.05** | +0.068 | +0.060 | **+0.042** | **+0.028** | **+0.010** | **✓** |
| 0.08 | +0.066 | +0.060 | +0.041 | +0.038 | −0.003 | ✗ |
| 0.12 | +0.057 | +0.054 | +0.037 | +0.040 | +0.003 | ✓ |
| 0.15（原值） | +0.036 | +0.034 | +0.032 | +0.040 | −0.002 | ✗ |

默认值 0.15 在大面板上只有最优值的一半。决策规则取"每个 k 都非负" → **0.05**。

---

## 6. 逐类诊断：分数丢在哪里

用最强的 CPU 预测器（4-view SVC + soft centring），full-OOF bacc 0.4415 / board-like 0.4408。

**家族结构**：

| 家族 | 类数 | 家族召回 | 家族内条件正确率 | 整体召回 |
|---|---:|---:|---:|---:|
| anthropic | 5 | 0.717 | 0.461 | 0.461 |
| google | 4 | 0.993 | 0.588 | 0.588 |
| openai | 5 | 0.881 | 0.455 | 0.455 |
| zai | 2 | 0.387 | 0.200 | 0.200 |
| **moonshot** | 1 | — | — | **0.040** |
| qwen | 1 | — | — | 0.580 |

**错误中 53.8% 是同族内部**（811/1507）。

**可挽回分数最多（points available）的类**：
`kimi-k3`（召回 0.04）、`claude-fable-5`（0.12）、`glm-5.3`、`gpt-5.6-sol`、`glm-5.3-flash`。
仅把 `kimi-k3` 从 0.04 提到 0.70，就是 **+0.037**。

**主要混淆对**：`gemini-3.8↔3.7`(63/39)、`glm-5.3→claude-opus-5`(44)、`kimi-k3→gpt-5.5`(43)、`gpt-5.6-sol→gpt-6-astra`(36)。

---

## 7. 为什么没进前十（结论）

**所有"重排已有信息"的改动都被噪声淹没。** 六个版本的实盘跨度 0.664–0.670（0.006），而公开实测的同代码多次提交极差是 0.009。

尝试过并**全部失败**的新信息源：

| 尝试 | 结论 |
|---|---|
| `id` 列是否泄漏 prompt 顺序 | 按 id 排序的连续同 prompt 段数 2680 vs 随机对照 2682±4.1（z=−0.54）→ id 是随机哈希 |
| Kaggle 上目标模型的真实权重 | 存在（Kimi-K3 GGUF、GLM-5.3 81GB 分片），但**不可加载**：前者是 llama.cpp 按层切片，后者只有 `blk_*.safetensors`，**无 config、无 tokenizer** |
| Qwen3-30B-A3B（MoE）做更强成员 | 已下载 57GB、改造训练器支持 MoE（LoRA 打 attn+router，冻结 128 专家），但 **15.4 s/step**（16 层仍 14.4s），比 14B 慢 10 倍，单次 117h → 放弃 |
| 自建生成数据 | 论坛多人证伪：系统提示/温度不匹配会引入模板偏差 |

**判断**：0.72–0.77 那批队伍掌握的是**显著更强的单模型**（或非公开的数据构造流程），而不是对公开 pool 的重新组合。在公开信息范围内，0.67 附近是这套方法的上限。

**下一步若继续，优先级**：(1) 打 `kimi-k3` / `claude-fable-5` / `glm-*` 这几个低召回类（§6 已给出可挽回分数）；(2) 用层次分类（先判家族、再判类内），因为 53.8% 错误是同族内；(3) 继续扩大独立语料（n-gram 侧未见饱和）。

---

## 8. 仓库结构

```
corpus/          验证协议与语料构建
  cv.py            唯一验证真源（LABELS / prompt_folds / sim_subset / evaluate / board_like_score）
  build_corpus.py  pool 并集构建（去重 / 长度过滤 / 按类封顶）
train/           两阶段截断 LoRA（Ascend NPU）
  lora24.py        共享核心：截断、手写 LoRA、8 层池化、双头、checkpoint
  train_lora.py    两阶段训练
  predict_lora.py  推理出 (n,18) log-prob
  oof_lora.py      逐折重训拿真 OOF
  run_card.sh      单卡启动（source Ascend 环境、校验 HBM、不足则 exit 90）
experiments/     所有结论对应的实验脚本
  exp_svc_scale.py      语料规模曲线
  exp_soft_trans.py     soft vs hard 转导
  exp_thr_sweep.py      adapt_thr 校准
  exp_partial_panel.py  残缺面板下的适用边界
  exp_kb_proxy.py       kb_proxy 是否污染
  diag_perclass.py      逐类诊断 / 家族结构 / 混淆矩阵
results/         聚合后的实验数字（JSON）
EXPERIMENT_LOG.md 完整台账：每个假设、每个否证、每条环境坑
```

---

## 9. 复现

```bash
# 1. 语料（需要自行下载公开 pool 数据集到 pools/<owner>__<slug>/）
python corpus/build_corpus.py --refresh

# 2. 验证协议自检
python corpus/cv.py

# 3. 关键实验（CPU，建议在 192 核机器上跑）
python experiments/exp_svc_scale.py --cfgs 0,3000,9000,0
python experiments/exp_soft_trans.py
python experiments/exp_thr_sweep.py

# 4. 训练（Ascend NPU；每张卡一个进程）
bash train/run_card.sh 3 --python -u train/train_lora.py \
  --model <Qwen3-14B-Base> --layers 27 --bs 8 --epochs1 1 --epochs2 2
```

---

## 10. 工程注意事项（实测踩过的坑）

- `sed -i 's/\r$//'` 在某些 GNU sed 上会把**行尾的字母 r 删除**（`\r` 被当作字面量 r），曾把 `AutoTokenizer` 变成 `AutoTokenize`。改用 Python 按字节替换 `\r\n`。
- Qwen3 依赖 `einops`；缺失时 `transformers` 的 Qwen3 类**和 peft** 同时导入失败，报错完全不相干。
- pandas3 + numpy2 下，按 numpy1.x 编译的 `numexpr`/`bottleneck` 被 import 会打出 C 级 traceback；用 `sys.modules.setdefault('numexpr', None)` 屏蔽。
- 任何 NPU 进程启动前必须 `source .../set_env.sh`，否则 `libhccl.so: cannot open shared object file`。
- `PYTHONPATH` 只能追加不能覆盖（覆盖会导致 `No module named 'tbe'`）。
- 多 adapter 成员：公开推理代码**不支持** `adapter_weights`，多个 adapter 会被等权平均——设权重是无效的。

# Communication Master

- **LLM 缓存策略学习**：SFT / GRPO / DAPO（基于 TRL + Unsloth）

## 目录结构

- `cache_env/`：多基站缓存仿真环境 + prompt/动作格式工具  
  - `unified_multi_bs_cache_env.py`：环境（状态、缓存、命中率、频率特征、step/reset 等）
  - `unified_data_loader.py`：数据生成/加载（Zipf、多用户等）
  - `prompt_utils.py`：把环境状态格式化为 LLM prompt（system/user messages）
  - `history_rebuilder.py`：SFT “教师策略”（lookahead/穷举）与相关工具
  - `action_validator.py`：严格解析/校验 LLM 输出动作格式（Replace/NoOp）

- `llm/`：训练与评估入口脚本  
  - `train_sft.py`：生成 SFT 数据（教师决策）并进行 LoRA SFT
  - `train_grpo.py`：GRPO 训练（奖励基于缓存命中率/增益等）
  - `train_dapo.py`：DAPO 训练（`GRPOConfig(loss_type="dapo")`）
  - `evaluate_unified.py`：统一评估脚本（可选，含多种评估逻辑兜底导入）

- `tools/`：辅助工具  
  - `download_hf_model.py`：从 Hugging Face 下载模型快照到本地目录
  - `merge_lora.py`：把 LoRA adapter 合并到 base model 并保存为 merged 模型

- `baselines/sac_baseline/`：SAC baseline（原仓库 README 保留在该目录）

## 环境安装

建议：Linux + NVIDIA GPU（Unsloth / bitsandbytes / vLLM 多数情况下只在 Linux 上可用）。

```bash
pip install -r requirements.txt
```

说明：
- `requirements.txt` 会包含 `requirements-llm.txt` + `requirements-sac.txt`。
- `torch` 建议按你的 CUDA 版本用官方方式安装（不同平台/驱动差异较大）。

## 依赖说明（核心库）

- `torch`：训练与推理计算
- `transformers`：Tokenizer / 模型加载 / generation 配置
- `trl`：`SFTTrainer`、`GRPOTrainer`、`GRPOConfig`
- `unsloth`：加速 LoRA/SFT/GRPO 训练（可选但推荐）
- `datasets`：训练数据集封装（`Dataset`）
- `numpy` / `pandas`：环境与数据生成/处理
- `tqdm`：进度条
- `huggingface_hub`：`tools/download_hf_model.py` 下载模型
- `tensorboard` / `wandb`：训练日志
- `gym`：`baselines/sac_baseline` 环境接口

## 下载模型

```bash
python tools/download_hf_model.py --repo_id Qwen/Qwen2.5-7B-Instruct --out models/Qwen2.5-7B-Instruct
```

## 推荐流程（LLM）

1) 下载基座模型（或准备本地模型目录）  
2) 运行 `llm/train_sft.py` 训练 LoRA（教师策略生成数据）  
3) 用 `tools/merge_lora.py` 合并 LoRA 得到一个 merged 模型目录（可选，但 GRPO/DAPO 通常更方便）  
4) 运行 `llm/train_grpo.py` 或 `llm/train_dapo.py` 做偏好/策略优化  

LoRA 合并示例：

```bash
python tools/merge_lora.py --base_model models/Qwen2.5-7B-Instruct --adapter outputs/sft/<run>/final_sft_checkpoint --out models/merge7B_exbert
```

## 运行（LLM 部分）

这些脚本都支持通过环境变量覆盖路径（默认写到仓库下的 `data/` 与 `outputs/`）：

- SFT：
  - `SFT_MODEL_PATH`：基座模型路径
  - `SFT_DATA_DIR`：SFT 数据输出目录
  - `SFT_OUTPUT_DIR`：SFT 训练输出目录

```bash
python llm/train_sft.py
```

- GRPO：
  - `GRPO_MODEL_PATH`：基座/merged 模型路径
  - `GRPO_DATA_ROOT`：GRPO 数据根目录
  - `GRPO_OUTPUT_ROOT`：GRPO 输出根目录
  - `USE_VLLM=1`：启用 vLLM（若未安装 vLLM 会自动回退为关闭）

```bash
python llm/train_grpo.py
```

- DAPO：
  - `DAPO_MODEL_PATH` / `DAPO_DATA_ROOT` / `DAPO_OUTPUT_ROOT` 同理

```bash
python llm/train_dapo.py
```

## 2 基站 / 5 基站：训练与超参数

本仓库的训练脚本里，**基站数/用户数等环境超参是写在脚本头部的全局常量**。目前默认值是：

- `llm/train_sft.py`：默认 **B=2**（`NUM_BASE_STATIONS=2, NUM_USERS=20`）
- `llm/train_grpo.py`：默认 **B=5**（`NUM_BASE_STATIONS=5, NUM_USERS=40`）
- `llm/train_dapo.py`：默认 **B=2**（`NUM_BASE_STATIONS=2, NUM_USERS=20`）
- `llm/evaluate_unified.py`：**用命令行参数**切换 B=2 / B=5

下面把 **B=2** 和 **B=5** 的训练/评测流程、以及你需要的超参数都集中写一遍，照着做就能复现实验。

### 两基站（B=2）：训练

#### 1) SFT（教师策略数据 + LoRA SFT）

脚本：`llm/train_sft.py`

关键环境超参：

- `NUM_BASE_STATIONS=2`
- `NUM_USERS=20`
- `NUM_CONTENTS=100`
- `CACHE_SIZES=[10,10]`
- `ZIPF_PARAM=1.2`
- `NUM_EPOCHS_DATA_SFT=3000`（SFT 数据量）
- `MAX_SEQ_LENGTH=2048`

运行：

```bash
# （可选）指定基座模型目录（HF 格式）；不设则默认 models/Qwen2.5-7B-Instruct
export SFT_MODEL_PATH=models/Qwen2.5-7B-Instruct

python llm/train_sft.py
```

输出说明：

- SFT 数据：写入 `data/sft/`，文件名类似 `sft_data_no_cot_replacement_only_v1_nbs2_seed666.json`
- SFT LoRA：写入 `outputs/sft/SFT_Qwen2_Cache_<时间戳>/final_sft_checkpoint/`

#### 2) 合并 SFT LoRA，得到 merged 模型（便于 GRPO/DAPO）

```bash
python tools/merge_lora.py \
  --base_model models/Qwen2.5-7B-Instruct \
  --adapter outputs/sft/SFT_Qwen2_Cache_<时间戳>/final_sft_checkpoint \
  --out models/merge7B_exbert
```

#### 3) GRPO（B=2 版）

**如果你要训练 B=2 的 GRPO**，请把 `llm/train_grpo.py` 文件头部的全局常量改成下面这套：

```python
# ===== B=2（两基站）GRPO 超参数 =====
MAX_COMPLETION_LENGTH = 64
MAX_PROMPT_LENGTH = 2048
MAX_SEQ_LENGTH = MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH

LORA_RANK = 32
LORA_ALPHA = 32

NUM_BASE_STATIONS = 2
NUM_USERS = 20
NUM_CONTENTS = 100
CACHE_SIZES = [10, 10]
ZIPF_PARAM = 1.2
NUM_EPOCHS_FOR_GRPO_DATA = 10000
FUTURE_STEPS_REWARD = 10
FUTURE_STEPS_SAMPLING = 5

GRPO_DATA_SEED = 4567
TRAINER_SEED = 42

ANNEAL_STRATEGY = "cosine"
ANNEAL_TEMP_START = 0.90
ANNEAL_TEMP_END = 0.75
ANNEAL_TOP_P_START = 0.95
ANNEAL_TOP_P_END = 0.85

NUM_GENERATIONS = 4
TEMPERATURE = ANNEAL_TEMP_START
TOP_P = ANNEAL_TOP_P_START
TOP_K = 50
REPETITION_PENALTY = 1.0

LEARNING_RATE = 1e-4
PER_DEVICE_TRAIN_BATCH = 4
GRAD_ACCUM_STEPS = 12
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 1
SAVE_STEPS = 100
MAX_GRAD_NORM = 0.1

DEBUG_SAMPLING_SMOKE_TEST = True
PRINT_ALL_GROUPS = True
MAX_GROUPS_TO_PRINT = 3
SHOW_GLOBAL_BEST_WORST = True
RESET_CACHE_EVERY = 1500

REWARD_HIT_MODE = "delta_weighted"
GAIN_SCALE = 1.0

EARLY_PHASE_STEPS = 120
NOOP_PENALTY_LATE = 0.005
NOOP_PENALTY_EARLY_WITH_OPP = 0.0075
NOOP_PENALTY_EARLY_NO_OPP = 0.0025
MAX_JOINT_COMBOS_PER_STEP = 50000
UNCOND_NOOP_PENALTY_STEPS = 30
UNCOND_NOOP_PENALTY = 0.005

OPP_EVAL_STEPS = 5
OPP_DISCOUNT_GAMMA = 0.9

SLOT_INDEX_BASE = 0

ANNEAL_ENABLED = bool(int(os.getenv("ANNEAL_ENABLED", "1")))
ANNEAL_LOG_EVERY = int(os.getenv("ANNEAL_LOG_EVERY", "5"))
```

然后运行：

```bash
# （可选）关闭退火：ANNEAL_ENABLED=0；默认 ANNEAL_ENABLED=1 开启
export ANNEAL_ENABLED=1
export GRPO_MODEL_PATH=models/merge7B_exbert

python llm/train_grpo.py
```

输出说明：

- GRPO 数据：写入 `data/grpo/grpo_cache_data_<时间戳>/`
- GRPO LoRA：写入 `outputs/grpo/grpo_cache_<时间戳>/final_lora_weights/`

#### 4)（可选）DAPO（B=2 版）

脚本：`llm/train_dapo.py`（当前代码默认就是 B=2）

```bash
export DAPO_MODEL_PATH=models/merge7B_exbert
python llm/train_dapo.py
```

### 五基站（B=5）：训练

#### 1) SFT

**如果你要训练 B=5 的 SFT**，建议至少把以下常量改为：

- `NUM_BASE_STATIONS = 5`
- `NUM_USERS = 40`
- `CACHE_SIZES = [10,10,10,10,10]`
- `MAX_SEQ_LENGTH = 4096`（B=5 prompt 更长；过小会被过滤掉很多样本）

改完后照常运行：

```bash
export SFT_MODEL_PATH=models/Qwen2.5-7B-Instruct
python llm/train_sft.py
```

#### 2) 合并 SFT LoRA

同上，得到 `models/merge7B_exbert`（或你自定义的 merged 目录），供 GRPO 使用。

#### 3) GRPO（B=5 版）

**如果你要训练 B=5 的 GRPO**，请把 `llm/train_grpo.py` 文件头部的全局常量改成下面这套：

```python
# ===== B=5（五基站）GRPO 超参数（与当前代码一致）=====
MAX_COMPLETION_LENGTH = 128
MAX_PROMPT_LENGTH = 4096
MAX_SEQ_LENGTH = MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH

LORA_RANK = 32
LORA_ALPHA = 32

NUM_BASE_STATIONS = 5
NUM_USERS = 40
NUM_CONTENTS = 100
CACHE_SIZES = [10, 10, 10, 10, 10]
ZIPF_PARAM = 1.2
NUM_EPOCHS_FOR_GRPO_DATA = 10000
FUTURE_STEPS_REWARD = 10
FUTURE_STEPS_SAMPLING = 5

GRPO_DATA_SEED = 4567
TRAINER_SEED = 42

ANNEAL_STRATEGY = "cosine"
ANNEAL_TEMP_START = 0.90
ANNEAL_TEMP_END = 0.75
ANNEAL_TOP_P_START = 0.95
ANNEAL_TOP_P_END = 0.85

NUM_GENERATIONS = 4
TEMPERATURE = ANNEAL_TEMP_START
TOP_P = ANNEAL_TOP_P_START
TOP_K = 50
REPETITION_PENALTY = 1.0

LEARNING_RATE = 1e-4
PER_DEVICE_TRAIN_BATCH = 4
GRAD_ACCUM_STEPS = 12
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 1
SAVE_STEPS = 100
MAX_GRAD_NORM = 0.1

DEBUG_SAMPLING_SMOKE_TEST = True
PRINT_ALL_GROUPS = True
MAX_GROUPS_TO_PRINT = 3
SHOW_GLOBAL_BEST_WORST = True
RESET_CACHE_EVERY = 1500

REWARD_HIT_MODE = "delta_weighted"
GAIN_SCALE = 1.0

EARLY_PHASE_STEPS = 120
NOOP_PENALTY_LATE = 0.0075
NOOP_PENALTY_EARLY_WITH_OPP = 0.01
NOOP_PENALTY_EARLY_NO_OPP = 0.005
MAX_JOINT_COMBOS_PER_STEP = 50000
UNCOND_NOOP_PENALTY_STEPS = 30
UNCOND_NOOP_PENALTY = 0.0075

OPP_EVAL_STEPS = 5
OPP_DISCOUNT_GAMMA = 0.9

SLOT_INDEX_BASE = 0

ANNEAL_ENABLED = bool(int(os.getenv("ANNEAL_ENABLED", "1")))
ANNEAL_LOG_EVERY = int(os.getenv("ANNEAL_LOG_EVERY", "5"))
```

运行：

```bash
export GRPO_MODEL_PATH=models/merge7B_exbert
python llm/train_grpo.py
```

## 统一评估（2 基站 / 5 基站）

统一评估入口脚本：`llm/evaluate_unified.py`  
它会在**同一份冻结的数据**（相同 requests & 连接关系）上评估多种策略，并把结果写成 JSON。

常用参数：
- `--num_base_stations`：基站数量（2 或 5）
- `--cache_sizes`：每个基站 cache 大小（逗号分隔）。例如 `10,10` 或 `10,10,10,10,10`
- `--num_contents`：内容数（默认 100）
- `--num_users_list`：要测的用户数列表（逗号分隔）。例如只测 40 个用户就写 `40`
- `--sac_ckpt`：SAC baseline 的 checkpoint（`.pt`）或目录（会自动选择 `sac_final.pt`/最新 step），默认指向 `sac_baseline_B5/sac_final.pt`
- `--grpo_five_lora_dir` / `--grpo_ten_lora_dir`：如果你要一起评估 LLM（SFT/GRPO），需要给出 LoRA 目录

输出位置：
- 结果会写到 `outputs/evaluation_outputs/`（可用 `--output_dir` 覆盖）
- 文件名会自动带上用户数量后缀，例如 `evaluation_results_users40.json`

说明：
- 当 `--num_base_stations > 2` 时，脚本会自动跳过 “单步穷举 / 三步 / 五步尾部NoOp” 这类穷举基线（因为组合数爆炸且原实现只对 2 基站有意义）。
- 若本机未安装 `unsloth/transformers`，脚本会自动跳过 LLM 评估，仅评估 SAC + 启发式（LRU/LFU/FIFO）。

### 测 2 个基站（B=2）

1) 准备（可选）LLM：
- `EVAL_MODEL_PATH` 指向你的 base/merged 模型目录（Hugging Face 格式）
- `--grpo_five_lora_dir` / `--grpo_ten_lora_dir` 指向对应的 LoRA adapter 目录

2) 准备 SAC checkpoint（如果你要对比 SAC）：
- 假设你的权重在 `sac_baseline_B2/sac_final.pt`（没有的话就用 `--sac_ckpt` 指定实际路径）

3) 运行示例（只测 20 用户）：

```bash
python llm/evaluate_unified.py --output eval_b2.json --num_base_stations 2 --cache_sizes 10,10 --num_contents 100 --num_users_list 20 --sac_ckpt sac_baseline_B2/sac_final.pt
```

### 测 5 个基站（B=5）

你当前已有的 SAC 权重目录：`sac_baseline_B5/`（包含 `sac_final.pt`、以及若干 `sac_step_*.pt`）

运行示例（只测 40 用户）：

```bash
python llm/evaluate_unified.py --output eval_b5.json --num_base_stations 5 --cache_sizes 10,10,10,10,10 --num_contents 100 --num_users_list 40 --sac_ckpt sac_baseline_B5
```

如果你也要同时评估 LLM（SFT/GRPO），在上面命令里追加：
- `--grpo_five_lora_dir <你的GRPO_FIVE适配器目录>`
- `--grpo_ten_lora_dir <你的GRPO_TEN适配器目录>`

## LLM 动作格式

LLM 的输出会被 `cache_env/action_validator.py` **严格解析**。默认只允许两类动作（每个基站一行）：

- 替换：`基站X的决策是：用内容Y替换槽位Z中的内容W。`
- 不操作：`基站X的决策是：不执行任何缓存操作。`

如果输出格式不满足约束（行数不对、ID 越界、替换内容不来自当前请求等），该 sample 会被判为无效并走兜底逻辑。

## 运行（SAC baseline）

在 `baselines/sac_baseline/` 下按其 README 运行即可，例如：

```bash
cd baselines/sac_baseline
python main.py --automatic_entropy_tuning True --target_update_interval 1000 --lr 1e-4 --exp-case case3 --cuda
```

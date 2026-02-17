# CoopLLM-Cache

* **LLM Cache Policy Learning**: SFT / GRPO / DAPO (based on TRL + Unsloth)

## Directory Structure

* `cache_env/`: Multi-base-station caching simulation environment + prompt/action-format utilities

  * `unified_multi_bs_cache_env.py`: Environment (state, cache, hit rate, frequency features, step/reset, etc.)
  * `unified_data_loader.py`: Data generation/loading (Zipf, multi-user, etc.)
  * `prompt_utils.py`: Formats environment state into LLM prompts (system/user messages)
  * `history_rebuilder.py`: SFT “teacher policy” (lookahead/exhaustive search) and related tools
  * `action_validator.py`: Strictly parses/validates LLM action output format (Replace/NoOp)

* `llm/`: Training and evaluation entry scripts

  * `train_sft.py`: Generate SFT data (teacher decisions) and run LoRA SFT
  * `train_grpo.py`: GRPO training (rewards based on cache hit rate/gain, etc.)
  * `train_dapo.py`: DAPO training (`GRPOConfig(loss_type="dapo")`)
  * `evaluate_unified.py`: Unified evaluation script (optional; includes multiple evaluation logic fallbacks via defensive imports)

* `tools/`: Helper tools

  * `download_hf_model.py`: Download Hugging Face model snapshots to a local directory
  * `merge_lora.py`: Merge a LoRA adapter into the base model and save as a merged model

* `baselines/sac_baseline/`: SAC baseline (original repo README preserved in this directory)

## Environment Setup

Recommended: Linux + NVIDIA GPU (Unsloth / bitsandbytes / vLLM are mostly only available on Linux in many cases).

```bash
pip install -r requirements.txt
```

Notes:

* `requirements.txt` includes `requirements-llm.txt` + `requirements-sac.txt`.
* For `torch`, it’s recommended to install via the official method based on your CUDA version (platform/driver differences can be significant).

## Dependencies (Core Libraries)

* `torch`: Training and inference computation
* `transformers`: Tokenizer / model loading / generation configuration
* `trl`: `SFTTrainer`, `GRPOTrainer`, `GRPOConfig`
* `unsloth`: Accelerated LoRA/SFT/GRPO training (optional but recommended)
* `datasets`: Training dataset wrapper (`Dataset`)
* `numpy` / `pandas`: Environment and data generation/processing
* `tqdm`: Progress bar
* `huggingface_hub`: Used by `tools/download_hf_model.py` to download models
* `tensorboard` / `wandb`: Training logs
* `gym`: Environment interface for `baselines/sac_baseline`

## Download the Model

```bash
python tools/download_hf_model.py --repo_id Qwen/Qwen2.5-7B-Instruct --out models/Qwen2.5-7B-Instruct
```

## Recommended Workflow (LLM)

1. Download the base model (or prepare a local model directory)
2. Run `llm/train_sft.py` to train LoRA via SFT (teacher policy generates data)
3. Use `tools/merge_lora.py` to merge LoRA into the base model and produce a merged model directory (optional, but GRPO/DAPO is usually more convenient this way)
4. Run `llm/train_grpo.py` or `llm/train_dapo.py` for preference/policy optimization

LoRA merge example:

```bash
python tools/merge_lora.py --base_model models/Qwen2.5-7B-Instruct --adapter outputs/sft/<run>/final_sft_checkpoint --out models/merge7B_exbert
```

## Running (LLM Part)

All these scripts support overriding paths via environment variables (defaults write into `data/` and `outputs/` under the repo):

* SFT:

  * `SFT_MODEL_PATH`: Base model path
  * `SFT_DATA_DIR`: SFT data output directory
  * `SFT_OUTPUT_DIR`: SFT training output directory

```bash
python llm/train_sft.py
```

* GRPO:

  * `GRPO_MODEL_PATH`: Base/merged model path
  * `GRPO_DATA_ROOT`: GRPO data root directory
  * `GRPO_OUTPUT_ROOT`: GRPO output root directory
  * `USE_VLLM=1`: Enable vLLM (if vLLM is not installed, it will automatically fall back to disabled)

```bash
python llm/train_grpo.py
```

* DAPO:

  * `DAPO_MODEL_PATH` / `DAPO_DATA_ROOT` / `DAPO_OUTPUT_ROOT` are analogous

```bash
python llm/train_dapo.py
```

## 2 Base Stations / 5 Base Stations: Training and Hyperparameters

In this repo’s training scripts, **the number of base stations / number of users / and other environment hyperparameters are defined as global constants at the top of the scripts**. The current default values are:

* `llm/train_sft.py`: default **B=2** (`NUM_BASE_STATIONS=2, NUM_USERS=20`)
* `llm/train_grpo.py`: default **B=5** (`NUM_BASE_STATIONS=5, NUM_USERS=40`)
* `llm/train_dapo.py`: default **B=2** (`NUM_BASE_STATIONS=2, NUM_USERS=20`)
* `llm/evaluate_unified.py`: switch B=2 / B=5 via **command-line arguments**

Below is a consolidated write-up of the **B=2** and **B=5** training/evaluation workflow and the hyperparameters you need. Follow it as-is to reproduce the experiments.

### Two Base Stations (B=2): Training

#### 1) SFT (Teacher-Policy Data + LoRA SFT)

Script: `llm/train_sft.py`

Key environment hyperparameters:

* `NUM_BASE_STATIONS=2`
* `NUM_USERS=20`
* `NUM_CONTENTS=100`
* `CACHE_SIZES=[10,10]`
* `ZIPF_PARAM=1.2`
* `NUM_EPOCHS_DATA_SFT=3000` (SFT data volume)
* `MAX_SEQ_LENGTH=2048`

Run:

```bash
# (Optional) Specify the base model directory (HF format); if not set, defaults to models/Qwen2.5-7B-Instruct
export SFT_MODEL_PATH=models/Qwen2.5-7B-Instruct

python llm/train_sft.py
```

Output notes:

* SFT data: written to `data/sft/`, filenames like `sft_data_no_cot_replacement_only_v1_nbs2_seed666.json`
* SFT LoRA: written to `outputs/sft/SFT_Qwen2_Cache_<timestamp>/final_sft_checkpoint/`

#### 2) Merge SFT LoRA to Produce a Merged Model (Convenient for GRPO/DAPO)

```bash
python tools/merge_lora.py \
  --base_model models/Qwen2.5-7B-Instruct \
  --adapter outputs/sft/SFT_Qwen2_Cache_<timestamp>/final_sft_checkpoint \
  --out models/merge7B_exbert
```

#### 3) GRPO (B=2 Version)

**If you want to train GRPO with B=2**, change the global constants at the top of `llm/train_grpo.py` to the following:

```python
# ===== B=2 (Two Base Stations) GRPO Hyperparameters =====
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
ANNEAL_LOG_EVERY = int(os.getenv("ANNEAL_LOG_EVERY", "5")))
```

Then run:

```bash
# (Optional) Disable annealing: ANNEAL_ENABLED=0; by default ANNEAL_ENABLED=1 enables it
export ANNEAL_ENABLED=1
export GRPO_MODEL_PATH=models/merge7B_exbert

python llm/train_grpo.py
```

Output notes:

* GRPO data: written to `data/grpo/grpo_cache_data_<timestamp>/`
* GRPO LoRA: written to `outputs/grpo/grpo_cache_<timestamp>/final_lora_weights/`

#### 4) (Optional) DAPO (B=2 Version)

Script: `llm/train_dapo.py` (current code defaults to B=2)

```bash
export DAPO_MODEL_PATH=models/merge7B_exbert
python llm/train_dapo.py
```

### Five Base Stations (B=5): Training

#### 1) SFT

**If you want to train SFT with B=5**, it is recommended to at least change the following constants:

* `NUM_BASE_STATIONS = 5`
* `NUM_USERS = 40`
* `CACHE_SIZES = [10,10,10,10,10]`
* `MAX_SEQ_LENGTH = 4096` (B=5 prompts are longer; if too small, many samples will be filtered out)

After changes, run as usual:

```bash
export SFT_MODEL_PATH=models/Qwen2.5-7B-Instruct
python llm/train_sft.py
```

#### 2) Merge SFT LoRA

Same as above: produce `models/merge7B_exbert` (or a custom merged directory) for GRPO.

#### 3) GRPO (B=5 Version)

**If you want to train GRPO with B=5**, change the global constants at the top of `llm/train_grpo.py` to the following:

```python
# ===== B=5 (Five Base Stations) GRPO Hyperparameters (matches current code) =====
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
ANNEAL_LOG_EVERY = int(os.getenv("ANNEAL_LOG_EVERY", "5")))
```

Run:

```bash
export GRPO_MODEL_PATH=models/merge7B_exbert
python llm/train_grpo.py
```

## Unified Evaluation (2 Base Stations / 5 Base Stations)

Unified evaluation entry script: `llm/evaluate_unified.py`
It evaluates multiple strategies on the **same frozen dataset** (same requests & connectivity) and writes results as JSON.
By default, evaluation uses greedy decoding.

Common arguments:

* `--num_base_stations`: number of base stations (2 or 5)
* `--cache_sizes`: cache size per base station (comma-separated), e.g. `10,10` or `10,10,10,10,10`
* `--num_contents`: number of contents (default 100)
* `--num_users_list`: list of user counts to test (comma-separated), e.g. to test only 40 users use `40`
* `--sac_ckpt`: SAC baseline checkpoint (`.pt`) or directory (auto-selects `sac_final.pt` / latest step); defaults to `sac_baseline_B5/sac_final.pt`
* `--grpo_five_lora_dir` / `--grpo_ten_lora_dir`: if you want to evaluate LLMs (SFT/GRPO) together, provide the LoRA directories
* `--num_steps`: number of steps per evaluation (default 300)
* `--num_seeds`: number of seeds to average (default 3)

Output location:

* Results are written to `outputs/evaluation_outputs/` (can be overridden with `--output_dir`)
* Filenames automatically include the user-count suffix, e.g. `evaluation_results_users40.json`

### Evaluate 2 Base Stations (B=2)

1. Prepare (optional) LLM:

* `EVAL_MODEL_PATH` points to your base/merged model directory (Hugging Face format)
* `--grpo_five_lora_dir` / `--grpo_ten_lora_dir` point to the corresponding LoRA adapter directories

2. Prepare SAC checkpoint (if you want SAC comparison):

* Suppose your weights are in `sac_baseline_B2/sac_final.pt` (if not, specify the actual path via `--sac_ckpt`)

3. Run example (test only 20 users):

```bash
python llm/evaluate_unified.py --output eval_b2.json --num_base_stations 2 --cache_sizes 10,10 --num_contents 100 --num_users_list 20 --sac_ckpt sac_baseline_B2/sac_final.pt
```

### Evaluate 5 Base Stations (B=5)

Your existing SAC weights directory: `sac_baseline_B5/` (contains `sac_final.pt` and multiple `sac_step_*.pt`)

Run example (test only 40 users):

```bash
python llm/evaluate_unified.py --output eval_b5.json --num_base_stations 5 --cache_sizes 10,10,10,10,10 --num_contents 100 --num_users_list 40 --sac_ckpt sac_baseline_B5
```

If you also want to evaluate LLM (SFT/GRPO) simultaneously, append to the above command:

* `--grpo_five_lora_dir <your GRPO_FIVE adapter directory>`
* `--grpo_ten_lora_dir <your GRPO_TEN adapter directory>`

## LLM Action Format

LLM outputs are **strictly parsed** by `cache_env/action_validator.py`. By default, only two types of actions are allowed (one line per base station):

* Replace: `Base station X’s decision is: use content Y to replace content W in slot Z.`
* No-op: `Base station X’s decision is: do not perform any caching operation.`

If the output format does not satisfy constraints (wrong number of lines, out-of-range IDs, replacement content not from the current request, etc.), that sample is considered invalid and will use fallback logic.

## Running (SAC Baseline)

Run according to the README under `baselines/sac_baseline/`, for example:

```bash
cd baselines/sac_baseline
python main.py --automatic_entropy_tuning True --target_update_interval 1000 --lr 1e-4 --exp-case case3 --cuda
```

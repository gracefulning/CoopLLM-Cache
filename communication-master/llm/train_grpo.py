# -*- coding: utf-8 -*-
# ----------------------------------------------------------------
# GRPO 训练脚本
# ----------------------------------------------------------------
import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 16384
dynamo.config.suppress_errors = True 

from unsloth import FastLanguageModel, is_bfloat16_supported
import numpy as np
import torch
import logging
import os
import sys
import json
import glob
import math
import random
from datetime import datetime
from typing import List, Any, Dict
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from transformers import GenerationConfig, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
from functools import lru_cache
import copy

# -------------------- Path setup (so imports work when run as a script) --------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 本地模块
from cache_env.unified_multi_bs_cache_env import UnifiedMultiBSCacheEnv
from cache_env.unified_data_loader import MultiUserDataLoaderZipf
from cache_env.action_validator import CacheActionValidator
from cache_env.history_rebuilder import get_sft_teaching_decision
from cache_env.prompt_utils import format_joint_state_to_llm_prompt

# ----------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------

_use_vllm_env = os.getenv("USE_VLLM", "0").strip().lower() in {"1", "true", "yes", "y"}
use_vllm_train = False
if _use_vllm_env:
    try:
        import vllm  # noqa: F401
        use_vllm_train = True
    except Exception:
        use_vllm_train = False

MODEL_PATH = os.getenv("GRPO_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "merge7B_exbert"))
OUTPUT_ROOT = os.getenv("GRPO_OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "outputs", "grpo"))
DATA_ROOT = os.getenv("GRPO_DATA_ROOT", os.path.join(PROJECT_ROOT, "data", "grpo"))

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
ANNEAL_TEMP_END   = 0.75
ANNEAL_TOP_P_START = 0.95
ANNEAL_TOP_P_END   = 0.85

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

# === Reward mode 开关 ===
# 可选: "delta_weighted" | "abs_weighted" | "delta_next" | "abs_next"
REWARD_HIT_MODE = "delta_weighted"
GAIN_SCALE = 1.0

# ====== 奖励塑形相关超参 ======
EARLY_PHASE_STEPS = 120
#NOOP_PENALTY_LATE  = 0.005
NOOP_PENALTY_LATE  = 0.0075

#NOOP_PENALTY_EARLY_WITH_OPP = 0.0075
NOOP_PENALTY_EARLY_WITH_OPP = 0.01

#NOOP_PENALTY_EARLY_NO_OPP   = 0.0025
NOOP_PENALTY_EARLY_NO_OPP   = 0.005

MAX_JOINT_COMBOS_PER_STEP = 50000
UNCOND_NOOP_PENALTY_STEPS = 30
#UNCOND_NOOP_PENALTY       = 0.005
UNCOND_NOOP_PENALTY       = 0.0075

# ====== NoOp 机会评估窗口 ======
OPP_EVAL_STEPS = 5
OPP_DISCOUNT_GAMMA = 0.9

# ====== 槽位编号基准 ======
SLOT_INDEX_BASE = 0

# === 退火回调开关（作为超参数） ===
# 直接改成 True/False；也支持用环境变量覆盖：ANNEAL_ENABLED=1 开启；ANNEAL_LOG_EVERY 控制日志频率
ANNEAL_ENABLED = bool(int(os.getenv("ANNEAL_ENABLED", "1")))  # 默认 0=关闭
ANNEAL_LOG_EVERY = int(os.getenv("ANNEAL_LOG_EVERY", "5"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- 可选：优化器回退 ----------------
try:
    import bitsandbytes as bnb  # noqa: F401
    _OPTIM = "adamw_8bit"
except Exception:
    _OPTIM = "adamw_torch"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("grpo_train_vanilla")
STEP_COUNTER = 0

# ----------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------

def convert_numpy_to_python_types(obj: Any) -> Any:
    if isinstance(obj, dict): return {k: convert_numpy_to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_to_python_types(elem) for elem in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    return obj

@lru_cache(maxsize=32768)
def _load_state_cached(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_state_readonly(path: str):
    return copy.deepcopy(_load_state_cached(path))

# ----------------------------------------------------------------
# 数据生成与加载
# ----------------------------------------------------------------

def generate_grpo_data(tokenizer: AutoTokenizer, data_dir: str) -> None:
    logger.info(f"正在 '{data_dir}' 中生成新的GRPO训练数据...")
    os.makedirs(data_dir, exist_ok=True)

    total_steps_needed = NUM_EPOCHS_FOR_GRPO_DATA + max(FUTURE_STEPS_REWARD, FUTURE_STEPS_SAMPLING) + 1
    data_loader = MultiUserDataLoaderZipf(NUM_CONTENTS, total_steps_needed, NUM_USERS, ZIPF_PARAM, seed=GRPO_DATA_SEED)

    env = UnifiedMultiBSCacheEnv(data_loader, NUM_BASE_STATIONS, CACHE_SIZES, NUM_USERS, NUM_CONTENTS)
    cache_validator = CacheActionValidator(NUM_CONTENTS, CACHE_SIZES)

    env.reset(reset_caches=True, reset_freqs=True)
    env.prefill_cache_with_expert_policy(get_sft_teaching_decision, cache_validator)

    pbar_max = NUM_EPOCHS_FOR_GRPO_DATA
    kept = 0
    for i in tqdm(range(pbar_max), desc="GRPO数据生成"):
        current_requests = None
        joint_actions_for_env_step = None
        try:
            if RESET_CACHE_EVERY and i > 0 and (i % RESET_CACHE_EVERY == 0):
                logger.info(f"在步骤 {i} 周期性重置缓存...")
                env.reset(reset_caches=True, reset_freqs=False)
                env.prefill_cache_with_expert_policy(get_sft_teaching_decision, cache_validator)

            if env.cur_epoch >= env.num_epochs - max(FUTURE_STEPS_REWARD, FUTURE_STEPS_SAMPLING):
                logger.warning(f"数据生成提前结束于步骤 {i}，因为未来数据不足。")
                break

            current_requests = env.requests[env.cur_epoch]
            requested_sets_t = env._get_requested_contents_per_bs(current_requests)

            allowed_sets = [set(map(int, requested_sets_t[b])) for b in range(env.num_base_stations)]
            allowed_contents_per_bs = [sorted(list(s)) for s in allowed_sets]

            need_plus_one = (env._last_freq_epoch < env.cur_epoch)
            t_extra_per_bs = [requested_sets_t[b] if need_plus_one else set() for b in range(env.num_base_stations)]

            messages_for_prompt = format_joint_state_to_llm_prompt(
                env,
                requested_sets_t,
                current_requests_sets_at_t=t_extra_per_bs
            )
            prompt_ids = tokenizer.apply_chat_template(
                messages_for_prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            if prompt_ids.shape[-1] > MAX_PROMPT_LENGTH:
                if need_plus_one:
                    env._update_frequency_features(current_requests)
                    env._last_freq_epoch = env.cur_epoch
                safe_actions = [env.NO_OP_ACTION_INDEX] * env.num_base_stations
                env.step(safe_actions)
                continue

            joint_actions_for_env_step = [
                get_sft_teaching_decision(env, bs_idx, requested_sets_t[bs_idx], cache_validator)
                for bs_idx in range(env.num_base_stations)
            ]

            future_requests_reward = [env.requests[env.cur_epoch + k] for k in range(1, FUTURE_STEPS_REWARD + 1)]
            future_requests_sampling = [env.requests[env.cur_epoch + k] for k in range(1, FUTURE_STEPS_SAMPLING + 1)]

            state_info = {
                "bs_caches": convert_numpy_to_python_types(env.bs_caches),
                "current_requests": convert_numpy_to_python_types(current_requests),
                "future_requests_reward": convert_numpy_to_python_types(future_requests_reward),
                "future_requests_sampling": convert_numpy_to_python_types(future_requests_sampling),
                "user_bs_connections": convert_numpy_to_python_types(env.user_bs_connections),
                "allowed_contents_per_bs": allowed_contents_per_bs,
            }
            main_data_path = os.path.join(data_dir, f"sample_{i}.json")
            state_info_path = os.path.join(data_dir, f"state_info_{i}.json")
            main_data = {
                "prompt": tokenizer.apply_chat_template(messages_for_prompt, tokenize=False, add_generation_prompt=True),
                "completion": "",
                "state_info_path": state_info_path
            }
            with open(main_data_path, 'w', encoding='utf-8') as f:
                json.dump(main_data, f, ensure_ascii=False, indent=2)
            with open(state_info_path, 'w', encoding='utf-8') as f:
                json.dump(state_info, f, ensure_ascii=False, indent=2)
            kept += 1

            if need_plus_one:
                env._update_frequency_features(current_requests)
                env._last_freq_epoch = env.cur_epoch
            env.step(joint_actions_for_env_step)

        except Exception as e:
            logger.error(f"GRPO数据生成步骤 {i} 出错: {e}")
            try:
                if env._last_freq_epoch < env.cur_epoch:
                    if current_requests is None:
                        current_requests = env.requests[env.cur_epoch]
                    env._update_frequency_features(current_requests)
                    env._last_freq_epoch = env.cur_epoch

                if isinstance(joint_actions_for_env_step, list) and len(joint_actions_for_env_step) == env.num_base_stations:
                    env.step(joint_actions_for_env_step)
                else:
                    env.step([env.NO_OP_ACTION_INDEX] * env.num_base_stations)
            except Exception:
                pass
            continue

    logger.info(f"GRPO训练数据已生成于 '{data_dir}'（有效样本 {kept} 条）。")


def load_and_prepare_dataset(tokenizer: AutoTokenizer, data_dir: str) -> Dataset:
    if not os.path.exists(data_dir) or not glob.glob(os.path.join(data_dir, "sample_*.json")):
        generate_grpo_data(tokenizer, data_dir)
    logger.info(f"正在从 '{data_dir}' 加载GRPO数据...")
    main_data_files = sorted(glob.glob(os.path.join(data_dir, "sample_*.json")))
    records = [json.load(open(f, 'r', encoding='utf-8')) for f in main_data_files]
    if not records:
        raise RuntimeError(f"在 '{data_dir}' 中未找到有效数据。")

    logger.info(f"数据加载完成，共 {len(records)} 条样本。跳过机会感知采样。")
    return Dataset.from_list(records)

# ----------------------------------------------------------------
# 奖励及采样辅助函数
# ----------------------------------------------------------------

def _calculate_hit_rate(cache_states: List[List[int]],
                        requests: List[int],
                        user_bs_connections: List[List[int]]) -> float:
    total_hits, total_requests = 0, 0
    cache_sets = [set(item for item in bs_cache if item != -1) for bs_cache in cache_states]

    for user_id, content_id in enumerate(requests):
        if content_id == -1:
            continue
        total_requests += 1
        is_hit = False
        if 0 <= user_id < len(user_bs_connections):
            for bs_idx in user_bs_connections[user_id]:
                if content_id in cache_sets[bs_idx]:
                    is_hit = True
                    break
        if is_hit:
            total_hits += 1

    return total_hits / total_requests if total_requests > 0 else 0.0


def _weighted_avg_hit_rate(caches, future_requests, user_bs_connections, gamma=1):
    if not future_requests:
        return 0.0
    weights = [gamma ** k for k in range(len(future_requests))]
    wsum = sum(weights) or 1.0
    return sum(_calculate_hit_rate(caches, reqs, user_bs_connections) * w for w, reqs in zip(weights, future_requests)) / wsum

# ----------------------------------------------------------------
# 奖励函数
# ----------------------------------------------------------------
_reward_validator: CacheActionValidator = None


def _get_reward_validator() -> CacheActionValidator:
    global _reward_validator
    if _reward_validator is None:
        _reward_validator = CacheActionValidator(NUM_CONTENTS, CACHE_SIZES)
    return _reward_validator


def caching_performance_reward_multi_step(
    completions: List[str],
    state_infos: List[Dict],
    early_phase: bool = False,
    uncond_noop_phase: bool = False,
) -> List[Dict[str, Any]]:
    results = []
    validator = _get_reward_validator()

    def _enumerate_actions_for_bs(cache: List[int], allowed_contents: List[int]) -> List[int]:
        actions = [0]  # 0 is NoOp
        if not allowed_contents or not cache:
            return actions
        for slot in range(len(cache)):
            for cid in allowed_contents:
                if cache[slot] == cid:
                    continue
                actions.append(slot * NUM_CONTENTS + cid + 1)
        return actions

    def _apply_action_to_cache(cache: List[int], action_idx: int) -> List[int]:
        new_cache = list(cache)
        if action_idx <= 0:
            return new_cache
        slot, content = divmod(action_idx - 1, NUM_CONTENTS)
        if 0 <= slot < len(new_cache):
            new_cache[slot] = content
        return new_cache

    def _precompute_enum(bs_caches_before, allowed_sets_str, opp_eval_seq, user_bs_connections):
        actions_per_bs: List[List[int]] = []
        for bs in range(NUM_BASE_STATIONS):
            actions_per_bs.append(_enumerate_actions_for_bs(bs_caches_before[bs], list(allowed_sets_str[bs])))

        total_combos = 1
        for a in actions_per_bs:
            total_combos *= max(1, len(a))
        if total_combos > MAX_JOINT_COMBOS_PER_STEP:
            return False, None, None, None

        if len(opp_eval_seq) == 1:
            next_noop = _calculate_hit_rate(bs_caches_before, opp_eval_seq[0], user_bs_connections)
        else:
            next_noop = _weighted_avg_hit_rate(bs_caches_before, opp_eval_seq, user_bs_connections, gamma=OPP_DISCOUNT_GAMMA)

        best_with_write = [next_noop for _ in range(NUM_BASE_STATIONS)]

        # 通用多基站枚举：迭代所有联合动作组合
        def _enumerate_joint_actions(index=0, current_actions=None):
            if current_actions is None:
                current_actions = []

            if index == NUM_BASE_STATIONS:
                # 已构建完整的联合动作，计算得分
                caches_tmp = [_apply_action_to_cache(bs_caches_before[bs], current_actions[bs])
                             for bs in range(NUM_BASE_STATIONS)]
                score = _calculate_hit_rate(caches_tmp, opp_eval_seq[0], user_bs_connections) if len(opp_eval_seq) == 1 \
                        else _weighted_avg_hit_rate(caches_tmp, opp_eval_seq, user_bs_connections, gamma=OPP_DISCOUNT_GAMMA)

                # 更新每个基站的最佳写入得分
                for bs in range(NUM_BASE_STATIONS):
                    if current_actions[bs] != 0:
                        best_with_write[bs] = max(best_with_write[bs], score)
                return

            # 递归枚举当前基站的所有动作
            for action in actions_per_bs[index]:
                _enumerate_joint_actions(index + 1, current_actions + [action])

        _enumerate_joint_actions()

        opp_by_enum = [best_with_write[b] > next_noop + 1e-6 for b in range(NUM_BASE_STATIONS)]
        return True, opp_by_enum, next_noop, best_with_write

    for i, completion in enumerate(completions):
        penalty_reasons = []
        try:
            parsed_actions = validator.parse_to_actions_or_none(completion)
            if parsed_actions is None:
                penalty_reasons.append(f"[-0.200] 决策解析失败 (格式或语法错误)")
                results.append({"score": -0.2, "details": {"main_score": 0.0, "penalty": -0.2, "reasons": penalty_reasons}})
                continue

            st = state_infos[i]
            bs_caches_before    = st.get("bs_caches")
            user_bs_connections = st.get("user_bs_connections")
            future_requests     = st.get("future_requests_reward")
            current_requests    = st.get("current_requests")
            allowed_sets_str    = st.get("allowed_contents_per_bs")

            if not all([bs_caches_before, user_bs_connections, future_requests, current_requests, allowed_sets_str]):
                penalty_reasons.append(f"[-0.200] 状态信息文件缺失关键字段")
                results.append({"score": -0.2, "details": {"main_score": 0.0, "penalty": -0.2, "reasons": penalty_reasons}})
                continue

            allowed_sets = [set(s) for s in allowed_sets_str]
            cache_sets_before = [set(x for x in c if x != -1) for c in bs_caches_before]
            next_req = future_requests[0] if future_requests else []

            next_req_per_bs = [set() for _ in range(NUM_BASE_STATIONS)]
            for u, cid in enumerate(next_req):
                if cid == -1: continue
                for bs in user_bs_connections[u]:
                    next_req_per_bs[bs].add(int(cid))

            window_req_per_bs = [set() for _ in range(NUM_BASE_STATIONS)]
            opp_window = future_requests[:max(1, int(OPP_EVAL_STEPS))]
            for step_reqs in opp_window:
                for u, cid in enumerate(step_reqs):
                    if cid == -1: continue
                    for bs in user_bs_connections[u]:
                        window_req_per_bs[bs].add(int(cid))

            opp_eval_seq = [next_req] if OPP_EVAL_STEPS <= 1 else opp_window

            enum_ok, opp_by_enum, next_noop_score, best_with_write = _precompute_enum(
                bs_caches_before, allowed_sets_str, opp_eval_seq, user_bs_connections
            )

            caches_after = [list(c) for c in bs_caches_before]
            penalty = 0.0
            noop_bs_seen, valid_replace_seen, invalid_replace_seen = set(), set(), set()

            def _apply_noop_penalty_for_bs(bs_id: int, reasons: List[str]):
                nonlocal penalty
                pen_val = 0.0
                reason_text = ""
                if uncond_noop_phase:
                    pen_val = UNCOND_NOOP_PENALTY
                    reason_text = f"BS {bs_id}: 无条件NoOp惩罚 (前{UNCOND_NOOP_PENALTY_STEPS}步)"
                else:
                    if enum_ok:
                        opp = bool(opp_by_enum[bs_id])
                        best_gain = max(0.0, float(best_with_write[bs_id] - next_noop_score)) if best_with_write and next_noop_score is not None else 0.0
                        scale = 1.5 if best_gain >= 0.05 else (1.2 if best_gain >= 0.02 else 1.0)
                    else:
                        allowed_set = allowed_sets[bs_id] if bs_id < len(allowed_sets) else set()
                        if OPP_EVAL_STEPS <= 1:
                            next_gap = next_req_per_bs[bs_id] - cache_sets_before[bs_id]
                        else:
                            next_gap = window_req_per_bs[bs_id] - cache_sets_before[bs_id]
                        opp = bool(next_gap & allowed_set)
                        scale = 1.0

                    if early_phase:
                        pen_val = (NOOP_PENALTY_EARLY_WITH_OPP if opp else NOOP_PENALTY_EARLY_NO_OPP) * scale
                        reason_text = f"BS {bs_id}: 早期NoOp惩罚({'有机会' if opp else '无机会'})"
                    elif opp:
                        pen_val = NOOP_PENALTY_LATE * scale
                        reason_text = f"BS {bs_id}: 后期NoOp惩罚(有机会)"

                if pen_val > 0:
                    penalty -= pen_val
                    reasons.append(f"[-{pen_val:.4f}] {reason_text}")

            for action in parsed_actions:
                kind, params = action.get("action"), action.get("parameters", {})
                if kind == "CacheNoOp":
                    bs_id = int(params.get("station_id", -1))
                    if bs_id == -1:
                        penalty -= 0.03
                        penalty_reasons.append(f"[-0.030] NoOp: 基站ID无效 (-1)")
                        continue
                    if bs_id not in noop_bs_seen:
                        _apply_noop_penalty_for_bs(bs_id, penalty_reasons)
                        noop_bs_seen.add(bs_id)
                    continue

                if kind != "CacheReplace":
                    penalty -= 0.03
                    penalty_reasons.append(f"[-0.030] 动作类型未知: '{kind}'")
                    continue

                bs_id, s_raw, c, e = map(int, [params.get(k, -1) for k in ["station_id", "slot_index", "content_id", "content_to_evict_id"]])
                s0 = s_raw - 1 if SLOT_INDEX_BASE == 1 else s_raw
                is_valid = True

                if not (0 <= bs_id < NUM_BASE_STATIONS and 0 <= s0 < CACHE_SIZES[bs_id] and 0 <= c < NUM_CONTENTS and 0 <= e < NUM_CONTENTS):
                    is_valid = False
                    penalty -= 0.03
                    penalty_reasons.append(f"[-0.030] BS {bs_id}: 参数越界 (s0={s0}, c={c}, e={e})")

                if is_valid and bs_caches_before[bs_id][s0] != e:
                    penalty -= 0.01
                    penalty_reasons.append(f"[-0.010] BS {bs_id}: 驱逐ID不匹配 (槽位{s0}中实际为{bs_caches_before[bs_id][s0]}, 非{e})")

                if is_valid:
                    allowed_set = allowed_sets[bs_id] if bs_id < len(allowed_sets) else set()
                    if c not in allowed_set:
                        is_valid = False
                        penalty -= 0.03
                        penalty_reasons.append(f"[-0.030] BS {bs_id}: 写入的内容 {c} 不在允许列表")

                if is_valid and (c in cache_sets_before[bs_id]):
                    is_valid = False
                    penalty -= 0.03
                    penalty_reasons.append(f"[-0.030] BS {bs_id}: 写入的内容 {c} 已存在于缓存中")

                if is_valid:
                    caches_after[bs_id][s0] = c
                    valid_replace_seen.add(bs_id)
                else:
                    invalid_replace_seen.add(bs_id)

            for bs_id in range(NUM_BASE_STATIONS):
                if bs_id not in valid_replace_seen and bs_id not in noop_bs_seen and bs_id not in invalid_replace_seen:
                    _apply_noop_penalty_for_bs(bs_id, penalty_reasons)

            if REWARD_HIT_MODE == "delta_next":
                next_before = _calculate_hit_rate(bs_caches_before, next_req, user_bs_connections)
                next_after  = _calculate_hit_rate(caches_after,  next_req, user_bs_connections)
                main_score = next_after - next_before
            elif REWARD_HIT_MODE == "delta_weighted":
                base_avg_hit_rate   = _weighted_avg_hit_rate(bs_caches_before, future_requests, user_bs_connections)
                action_avg_hit_rate = _weighted_avg_hit_rate(caches_after,     future_requests, user_bs_connections)
                main_score = action_avg_hit_rate - base_avg_hit_rate
            elif REWARD_HIT_MODE == "abs_weighted":
                action_avg_hit_rate = _weighted_avg_hit_rate(caches_after, future_requests, user_bs_connections)
                main_score = action_avg_hit_rate
            elif REWARD_HIT_MODE == "abs_next":
                main_score = _calculate_hit_rate(caches_after,  next_req, user_bs_connections)
            else:
                base_avg_hit_rate   = _weighted_avg_hit_rate(bs_caches_before, future_requests, user_bs_connections)
                action_avg_hit_rate = _weighted_avg_hit_rate(caches_after,     future_requests, user_bs_connections)
                main_score = action_avg_hit_rate - base_avg_hit_rate

            raw_reward = GAIN_SCALE * main_score + penalty
            final_reward = float(np.clip(raw_reward, -1.0, 1.0))

            results.append({
                "score": final_reward,
                "details": {
                    "main_score": main_score,
                    "penalty": penalty,
                    "reasons": penalty_reasons
                }
            })

        except Exception:
            logging.exception("计算奖励时出错")
            penalty_reasons.append(f"[-0.250] 奖励函数内部出现异常（详见堆栈日志）")
            results.append({
                "score": -0.25,
                "details": {"main_score": 0.0, "penalty": -0.25, "reasons": penalty_reasons}
            })
    return results

# ----------------------------------------------------------------
# 其余部分
# ----------------------------------------------------------------

def reward_func(completions: List[str], prompts: List[str] = None, **kwargs: Any) -> List[float]:
    global STEP_COUNTER; STEP_COUNTER += 1
    paths = kwargs.get("state_info_path", [])
    num_generations = kwargs.get("num_generations", NUM_GENERATIONS)
    if not paths:
        return [-0.25] * len(completions)

    def pick_idx(i: int) -> int:
        if len(paths) == len(completions):
            return i
        elif len(paths) * num_generations == len(completions):
            return i // num_generations
        else:
            return min(i, len(paths) - 1)

    state_infos = [load_state_readonly(paths[pick_idx(i)]) for i in range(len(completions))]
    early_phase = STEP_COUNTER <= EARLY_PHASE_STEPS
    uncond_noop_phase = STEP_COUNTER <= UNCOND_NOOP_PENALTY_STEPS

    reward_outputs = caching_performance_reward_multi_step(
        completions,
        state_infos,
        early_phase=early_phase,
        uncond_noop_phase=uncond_noop_phase,
    )

    performance_scores = [r['score'] for r in reward_outputs]

    if performance_scores and (STEP_COUNTER % 5 == 0):
        print("\n" + "="*80, flush=True); print(f"** Log Step: {STEP_COUNTER} **", flush=True)
        print(f"奖励均值: {np.mean(performance_scores):.4f}", flush=True)
        if SHOW_GLOBAL_BEST_WORST:
            worst_idx, best_idx = int(np.argmin(performance_scores)), int(np.argmax(performance_scores))
            print("---------- 全局最差样本 ----------", flush=True)
            print(f"Completion:\n---\n{completions[worst_idx]}\n---", flush=True)
            print(f"得分: {performance_scores[worst_idx]:.4f}", flush=True)
            if best_idx != worst_idx:
                print("---------- 全局最佳样本 ----------", flush=True)
                print(f"Completion:\n---\n{completions[best_idx]}\n---", flush=True)
                print(f"得分: {performance_scores[best_idx]:.4f}", flush=True)

        group_summaries = []
        total = len(completions)
        group_count = (total + num_generations - 1) // num_generations
        for g in range(group_count):
            start, end = g * num_generations, min((g + 1) * num_generations, total)
            if start >= end: continue
            group_scores = performance_scores[start:end]
            rel_best_idx_in_group, rel_worst_idx_in_group = int(np.argmax(group_scores)), int(np.argmin(group_scores))
            group_summaries.append({
                "g": g, "spread": float(np.max(group_scores) - np.min(group_scores)),
                "best_idx_global": start + rel_best_idx_in_group,
                "worst_idx_global": start + rel_worst_idx_in_group,
            })
        group_summaries.sort(key=lambda x: x["spread"], reverse=True)

        pick_groups = group_summaries if PRINT_ALL_GROUPS else group_summaries[:MAX_GROUPS_TO_PRINT]
        for info in pick_groups:
            best_idx, worst_idx = info["best_idx_global"], info["worst_idx_global"]
            print("=" * 80, flush=True)
            print(f"[组内对比] Prompt组 g={info['g']} | spread={info['spread']:.6f}", flush=True)

            print("—— 组内最差 completion ——", flush=True)
            print(f"Completion:\n---\n{completions[worst_idx]}\n---", flush=True)
            print(f"得分: {performance_scores[worst_idx]:.4f}", flush=True)

            worst_details = reward_outputs[worst_idx].get("details", {})
            if worst_details:
                w_main = worst_details.get('main_score', 0.0)
                w_pen = worst_details.get('penalty', 0.0)
                w_reasons = worst_details.get('reasons', [])
                print(f"  [得分明细] 命中率增益: {w_main:.4f}, 总惩罚: {w_pen:.4f}", flush=True)
                if w_reasons:
                    print("  [扣分原因]:", flush=True)
                    for reason in w_reasons:
                        print(f"    - {reason}")

            print("—— 组内最佳 completion ——", flush=True)
            print(f"Completion:\n---\n{completions[best_idx]}\n---", flush=True)
            print(f"得分: {performance_scores[best_idx]:.4f}", flush=True)
        print("=" * 80 + "\n", flush=True)

    return performance_scores


def _anneal_value(start: float, end: float, ratio: float, strategy: str) -> float:
    r = max(0.0, min(1.0, ratio))
    if strategy == "linear": return start + (end - start) * r
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * r))


class GenParamAnnealCallback(TrainerCallback):
    def __init__(self, temp_start, temp_end, topp_start, topp_end, strategy, log_every=50):
        self.temp_start, self.temp_end = temp_start, temp_end
        self.topp_start = topp_start
        self.topp_end   = topp_end if topp_end is not None else topp_start
        self.strategy, self.log_every = strategy, log_every
        self._trainer = self._model = None

    def bind(self, trainer, model):
        self._trainer, self._model = trainer, model
        return self

    def _apply(self, state):
        max_steps = max(1, getattr(state, "max_steps", 1))
        ratio = min(1.0, float(state.global_step) / float(max_steps))
        new_temp = _anneal_value(self.temp_start, self.temp_end, ratio, self.strategy)
        new_topp = _anneal_value(self.topp_start, self.topp_end, ratio, self.strategy)

        # === vLLM 路径：更新 trainer.args（GRPOConfig）
        if self._trainer and hasattr(self._trainer, "args"):
            self._trainer.args.temperature = float(new_temp)
            self._trainer.args.top_p = float(new_topp)

        # === 非 vLLM 路径：保底同步 HF generate
        if self._trainer and getattr(self._trainer, "generation_config", None):
            self._trainer.generation_config.temperature = float(new_temp)
            self._trainer.generation_config.top_p = float(new_topp)
        if self._model and getattr(self._model, "generation_config", None):
            self._model.generation_config.temperature = float(new_temp)
            self._model.generation_config.top_p = float(new_topp)

        if self.log_every > 0 and state.global_step > 0 and state.global_step % self.log_every == 0:
            logger.info(f"[Anneal] Step={state.global_step}/{max_steps} | Temp={new_temp:.3f} | Top_p={new_topp:.3f}")

    def on_step_begin(self, args, state, control, **kwargs):
        self._apply(state)


# 可选：最小单元测试（默认不调用）
def _mini_test():
    caches = [[0, 1, 2], [3, 4, 5]]
    reqs = [0, 3, 7, -1]
    conns = [[0], [1], [0], [0]]
    assert abs(_calculate_hit_rate(caches, reqs, conns) - (2/3)) < 1e-9
    print("[mini_test] _calculate_hit_rate ok")


def main():
    # --- 随机种子 ---
    torch.manual_seed(TRAINER_SEED)
    np.random.seed(TRAINER_SEED)
    random.seed(TRAINER_SEED)
    set_seed(TRAINER_SEED)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_ROOT, f"grpo_cache_{ts}")
    data_dir = os.path.join(DATA_ROOT, f"grpo_cache_data_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("="*80 + "\n GRPO 训练开始 \n" + "="*80)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        fast_inference=use_vllm_train,
        load_in_4bit=True,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        device_map="auto",
    )
    logger.info("合并后的SFT基础模型已加载。")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model.config.use_cache = False
    except Exception:
        pass

    logger.info("正在为GRPO训练添加新的LoRA适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        use_gradient_checkpointing="unsloth",
        random_state=TRAINER_SEED,
    )
    logger.info("新的LoRA适配器已添加。")

    eos_ids = [t for t in {
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    } if isinstance(t, int) and t >= 0]

    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        max_new_tokens=MAX_COMPLETION_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
    )
    model.generation_config = gen_cfg  # 非 vLLM 路径会用到

    train_dataset = load_and_prepare_dataset(tokenizer, data_dir)

    training_args = GRPOConfig(
        use_vllm=use_vllm_train,
        vllm_mode="colocate",

        # === vLLM 采样参数必须写在这里 ===
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,

        learning_rate=LEARNING_RATE,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim=_OPTIM,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        report_to="tensorboard",
        gradient_checkpointing=True,
        output_dir=output_dir,
        remove_unused_columns=False,
        seed=TRAINER_SEED,
        scale_rewards=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,   # 使用 processing_class
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.generation_config = gen_cfg  # 非 vLLM 路径保底

    # === 使用开关控制是否注册退火回调 ===
    if ANNEAL_ENABLED:
        anneal_cb = GenParamAnnealCallback(
            temp_start=ANNEAL_TEMP_START, temp_end=ANNEAL_TEMP_END,
            topp_start=ANNEAL_TOP_P_START, topp_end=ANNEAL_TOP_P_END,
            strategy=ANNEAL_STRATEGY, log_every=ANNEAL_LOG_EVERY
        ).bind(trainer, model)
        trainer.add_callback(anneal_cb)
        logger.info(f"已启用退火回调：Temp {ANNEAL_TEMP_START:.3f}→{ANNEAL_TEMP_END:.3f}, "
                    f"Top_p {ANNEAL_TOP_P_START:.3f}→{ANNEAL_TOP_P_END:.3f}, strategy={ANNEAL_STRATEGY}")
    else:
        logger.info(f"退火回调已禁用：temperature/top_p 将固定为 {TEMPERATURE:.3f} / {TOP_P:.3f}")

    if DEBUG_SAMPLING_SMOKE_TEST:
        try:
            test_msgs = [{"role": "user", "content": "测试一下多样性；请随便回答两句中文。"}]
            test_ids = tokenizer.apply_chat_template(
                test_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            test_am = (test_ids != tokenizer.pad_token_id).long()
            with torch.inference_mode():
                outs = model.generate(
                    input_ids=test_ids,
                    attention_mask=test_am,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    repetition_penalty=REPETITION_PENALTY,
                    max_new_tokens=128,
                    num_return_sequences=NUM_GENERATIONS,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
            print("\n[DEBUG] 采样烟雾测试（应当出现多样输出）：")
            decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
            for i, s in enumerate(decoded):
                line = next(reversed(s.splitlines()), s[:120])
                print(f"  - 样本 {i+1}: {line[:180]}")
            print()
        except Exception as e:
            print(f"[DEBUG] 采样烟雾测试失败: {e}")

    logger.info("开始GRPO训练...")
    trainer.train()
    logger.info("GRPO训练完成。")

    final_lora_path = os.path.join(output_dir, "final_lora_weights")
    try:
        model.save_lora(final_lora_path)
    except Exception:
        model.save_pretrained(final_lora_path)
    tokenizer.save_pretrained(final_lora_path)
    logger.info(f"最终GRPO LoRA适配器已保存至: {os.path.abspath(final_lora_path)}")


if __name__ == "__main__":
    # _mini_test()
    main()

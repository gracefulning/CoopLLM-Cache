# -*- coding: utf-8 -*-
# 文件名: evaluate_unified.py

import os
import sys
import json
import gc
import random
import logging
import argparse
import glob
import re
from typing import List, Dict, Any, Tuple, Optional
import hashlib
import copy
import time  # 计时

# =========================
# 将项目根目录加入 Python 路径
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)
_logger.info(f"项目根目录 '{PROJECT_ROOT}' 已添加到 Python 搜索路径。")

# =========================
# 依赖
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
except Exception:
    FastLanguageModel = None  # type: ignore[assignment]

    def is_bfloat16_supported() -> bool:  # type: ignore[override]
        return False

try:
    from transformers import AutoTokenizer, set_seed, GenerationConfig
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoTokenizer = Any  # type: ignore[assignment]
    GenerationConfig = Any  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False

    def set_seed(seed: int) -> None:  # type: ignore[override]
        return None
from tqdm import tqdm

# =========================
# 本项目模块
# =========================
from cache_env.unified_multi_bs_cache_env import UnifiedMultiBSCacheEnv
from cache_env.unified_data_loader import MultiUserDataLoaderZipf
from cache_env.action_validator import CacheActionValidator
from cache_env.prompt_utils import format_joint_state_to_llm_prompt
from cache_env.history_rebuilder import get_sft_teaching_decision

# =========================
# 路径 & 全局常量
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = os.getenv(
    "EVAL_MODEL_PATH",
    os.getenv("GRPO_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "merge7B_exbert")),
)
OUTPUT_ROOT = os.getenv("EVAL_OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "outputs"))
DATA_ROOT = os.getenv("EVAL_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))

MAX_COMPLETION_LENGTH = 128
MAX_PROMPT_LENGTH     = 9000
MAX_SEQ_LENGTH        = MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH

NUM_BASE_STATIONS = 5
NUM_CONTENTS      = 100
CACHE_SIZES       = [10, 10,10 ,10 ,10]
ZIPF_PARAM        = 1.2 # 恢复为固定值

# === 定义 NUM_USERS 循环测试列表，并保留全局变量 NUM_USERS 作为当前值占位 ===
NUM_USERS_LIST_TO_TEST = [20, 40, 60, 80, 100, 120] # 在这里修改你需要测试的用户数量列表
NUM_USERS         = 40 # 这个值会在 main 循环中被动态更新

NUM_EPOCHS_FOR_GRPO_DATA = 10000
FUTURE_STEPS_REWARD   = 10
FUTURE_STEPS_SAMPLING = 5

GRPO_DATA_SEED = 6667
TRAINER_SEED   = 42

# 这些退火常量保留但贪婪/采样模式只用到固定取值
ANNEAL_STRATEGY    = "cosine"
ANNEAL_TEMP_START  = 0.90
ANNEAL_TEMP_END    = 0.75
ANNEAL_TOP_P_START = 0.95
ANNEAL_TOP_P_END   = 0.85

TOP_K              = 50
REPETITION_PENALTY = 1.0

# 采样评估默认超参（可选）
SAMPLE_TEMPERATURE = 0.90
SAMPLE_TOP_P       = 0.95
SAMPLE_TOP_K       = TOP_K

# —— 评估脚本自己的通用输出目录 ——
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "evaluation_outputs")

# —— LLM 适配器默认路径（与你的默认一致） ——
GRPO_FIVE_LORA_DIR_DEFAULT = ''
GRPO_TEN_LORA_DIR_DEFAULT  = "/root/autodl-tmp/grpo_cache_project_deeprl_multi/outputs/grpo_cache_outputs/grpo_cache_20251112_141836/checkpoint-800/"

# —— SAC baseline checkpoint 默认路径 ——
SAC_CKPT_DEFAULT = os.path.join(PROJECT_ROOT, "sac_baseline_B5", "sac_final.pt")

# =========================
# 评估策略及超参（默认贪婪，可选采样）
# =========================
NUM_EVALUATION_START_POINTS = 1
NUM_SIMULATION_STEPS        = 300

# 前缀平均命中率统计点
PREFIX_AVG_STEPS = [50, 100, 150, 200, 250, 300]

# 解码模式集合：默认只评估 greedy（可选加入采样）
DECODE_CHOICES = ["greedy"]

# 穷举安全阈值
MAX_JOINT_COMBOS_PER_STEP = 10000000000

# 贴现因子（用于 H 步尾部 NoOp 穷举评分）
DISCOUNT_GAMMA = 0.9

# =========================
# 辅助函数：清理/构造解码配置
# =========================
def _clean_model_generation_config(model):
    """将 sampling/beam 相关字段从模型的 generation_config 中清空，避免被自动合并进 generate 调用。"""
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    cfg.do_sample = False
    if hasattr(cfg, "num_beams"): cfg.num_beams = 1
    for k in ("temperature", "top_p", "top_k", "typical_p"):
        if hasattr(cfg, k):
            setattr(cfg, k, None)
    model.generation_config = cfg

def _build_greedy_gen_config(tokenizer, eos_ids):
    """显式传入干净的 GenerationConfig，确保不包含 sampling 字段，并与训练侧字段风格一致。"""
    gen_cfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=MAX_COMPLETION_LENGTH,
        repetition_penalty=float(REPETITION_PENALTY),
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=eos_ids,
    )
    # 彻底清空采样参数
    gen_cfg.temperature = None
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.typical_p = None
    return gen_cfg

def _build_sample_gen_config(tokenizer, eos_ids):
    """用于可选采样的配置。"""
    gen_cfg = GenerationConfig(
        do_sample=True,
        num_beams=1,
        max_new_tokens=MAX_COMPLETION_LENGTH,
        repetition_penalty=float(REPETITION_PENALTY),
        temperature=float(SAMPLE_TEMPERATURE),
        top_p=float(SAMPLE_TOP_P),
        top_k=int(SAMPLE_TOP_K),
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=eos_ids,
    )
    return gen_cfg

# =========================
# 工具函数
# =========================
def convert_numpy_to_python(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: convert_numpy_to_python(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_numpy_to_python(x) for x in data]
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data

def _compute_prefix_avg(hit_rates: List[float], prefix_steps: List[int]) -> Dict[str, float]:
    """计算前缀平均命中率（按指定步数）。"""
    res: Dict[str, float] = {}
    if not hit_rates:
        return res
    for t in prefix_steps:
        if len(hit_rates) >= t:
            res[str(t)] = float(np.mean(hit_rates[:t]))
    return res

# 允许选择是否重置频率（默认 False，避免每种策略的底层数据不同）
def _initialize_env_for_sample(env: UnifiedMultiBSCacheEnv, initial_epoch_id: int,
                               cache_validator: CacheActionValidator,
                               reset_freqs: bool = False):
    """使用专家策略回放历史至指定时间步。"""
    _logger.info(f"使用专家策略回放历史直至时间步 {initial_epoch_id}（reset_freqs={reset_freqs}）...")
    env.reset(reset_caches=True, reset_freqs=reset_freqs)
    while env.cur_epoch < initial_epoch_id:
        if env.cur_epoch >= env.num_epochs - 1:
            _logger.warning("数据用尽，提前结束历史回放。")
            break
        current_requests = env.requests[env.cur_epoch]
        requested_sets = env._get_requested_contents_per_bs(current_requests)
        joint_actions = []
        for bs_idx in range(env.num_base_stations):
            act_idx = get_sft_teaching_decision(env, bs_idx, requested_sets[bs_idx], cache_validator)
            joint_actions.append(int(act_idx))
        env.step(joint_actions)
    _logger.info(f"历史回放完成，当前时间步: {env.cur_epoch}")

def convert_parsed_action_to_index(parsed_actions: List[Dict[str, Any]],
                                   num_contents: int,
                                   num_base_stations: int) -> List[int]:
    """将解析后的动作转换为环境动作索引（不校验范围，由上层/新增sanitize负责）。"""
    indices = [0] * num_base_stations
    for act in parsed_actions:
        params = act.get("parameters", {})
        bs_id = params.get("station_id")
        if bs_id is None or not (0 <= bs_id < num_base_stations):
            continue
        if act.get("action") in ("CacheReplace", "CacheFill"):
            slot = params.get("slot_index")
            content = params.get("content_id")
            if slot is not None and content is not None:
                indices[bs_id] = slot * num_contents + content + 1
        # CacheNoOp -> 0
    return indices

# 对 LLM 解析出的动作做显式校验（越界则回退为 NoOp）
def sanitize_joint_actions(env: UnifiedMultiBSCacheEnv, joint_actions: List[int]) -> List[int]:
    safe = [0] * env.num_base_stations
    for bs in range(env.num_base_stations):
        a = joint_actions[bs] if bs < len(joint_actions) else 0
        if a <= 0:
            safe[bs] = 0
            continue
        slot, content = divmod(a - 1, env.num_contents)
        if 0 <= slot < len(env.bs_caches[bs]) and 0 <= content < env.num_contents:
            safe[bs] = a
        else:
            _logger.warning(f"[sanitize] 基站{bs} 动作越界(slot={slot}, content={content}) -> NoOp")
            safe[bs] = 0
    return safe

def _allowed_contents_per_bs_at(env: UnifiedMultiBSCacheEnv, epoch_index: int) -> List[List[int]]:
    """基于给定 epoch 的请求，映射到各基站的 allowed set。"""
    if epoch_index >= len(env.requests):
        return [[] for _ in range(env.num_base_stations)]
    reqs = env.requests[epoch_index]
    allowed_sets: List[set] = [set() for _ in range(env.num_base_stations)]
    for u, cid in enumerate(reqs):
        if cid == -1:
            continue
        if 0 <= u < len(env.user_bs_connections):
            for bs in env.user_bs_connections[u]:
                if 0 <= bs < env.num_base_stations:
                    allowed_sets[bs].add(int(cid))
    return [sorted(list(s)) for s in allowed_sets]

def _apply_action_to_cache(cache: List[int], action_idx: int, num_contents: int) -> List[int]:
    """返回应用 action 后的新 cache（不改原 cache）。"""
    new_cache = list(cache)
    if action_idx <= 0:
        return new_cache
    slot, content = divmod(action_idx - 1, num_contents)
    if 0 <= slot < len(new_cache):
        new_cache[slot] = content
    return new_cache

def _compute_instant_hit_rate(caches_after: List[List[int]],
                              requests_step: List[int],
                              user_bs_connections: List[List[int]]) -> float:
    """某一步的即时命中率。"""
    total_hits = 0
    total_requests = 0
    cache_sets = [set(x for x in c if x != -1) for c in caches_after]
    for user_id, cid in enumerate(requests_step):
        if cid == -1:
            continue
        total_requests += 1
        hit = False
        if 0 <= user_id < len(user_bs_connections):
            for bs in user_bs_connections[user_id]:
                if cid in cache_sets[bs]:
                    hit = True
                    break
        if hit:
            total_hits += 1
    return float(total_hits / total_requests) if total_requests > 0 else 0.0

def _enumerate_actions_for_bs(cache: List[int], allowed_contents: List[int],
                              num_contents: int) -> List[int]:
    """枚举某基站动作：0（no-op）+ 每槽×候选内容写入。"""
    actions = [0]
    if not allowed_contents or not cache:
        return actions
    for slot in range(len(cache)):
        for cid in allowed_contents:
            if cache[slot] == cid:
                continue
            actions.append(slot * num_contents + cid + 1)
    return actions

def _score_joint_action_single_step(env: UnifiedMultiBSCacheEnv, joint_actions: List[int]) -> float:
    """对候选联合动作计算下一步即时命中率得分（不改变 env）。"""
    caches_before = [list(c) for c in env.bs_caches]
    caches_after = []
    for bs in range(env.num_base_stations):
        a = joint_actions[bs] if bs < len(joint_actions) else 0
        caches_after.append(_apply_action_to_cache(caches_before[bs], a, env.num_contents))
    if env.cur_epoch + 1 >= len(env.requests):
        return 0.0
    next_reqs = env.requests[env.cur_epoch + 1]
    return _compute_instant_hit_rate(caches_after, next_reqs, env.user_bs_connections)

# ======== 解耦单步穷举（每个基站独立最大化下一步即时命中率） ========
def choose_exhaustive_joint_action_single_step(env: UnifiedMultiBSCacheEnv) -> List[int]:
    if env.cur_epoch >= len(env.requests) - 1:
        return [0] * env.num_base_stations

    allowed_per_bs = _allowed_contents_per_bs_at(env, env.cur_epoch)
    caches_before = [list(c) for c in env.bs_caches]
    requests_next = env.requests[env.cur_epoch + 1]
    user_bs_conns = env.user_bs_connections

    best_joint = [0] * env.num_base_stations
    for bs in range(env.num_base_stations):
        actions = _enumerate_actions_for_bs(caches_before[bs], allowed_per_bs[bs], env.num_contents)
        best_score = -1.0
        best_action = 0
        for a in actions:
            caches_after = [list(c) for c in caches_before]
            caches_after[bs] = _apply_action_to_cache(caches_before[bs], a, env.num_contents)
            score = _compute_instant_hit_rate(caches_after, requests_next, user_bs_conns)
            if score > best_score:
                best_score = score
                best_action = a
        best_joint[bs] = best_action

    return best_joint

# ======== H步（当前步枚举 + 后续全 NoOp）穷举：贴现平均 ========
def choose_exhaustive_joint_action_horizon_noop(env: UnifiedMultiBSCacheEnv, horizon_steps: int) -> List[int]:
    """
    在当前步枚举所有联合动作；随后 horizon_steps-1 步均 NoOp；
    打分：从 t+1 到 t+horizon_steps 的即时命中率做贴现平均（DISCOUNT_GAMMA^k）。
    """
    if env.cur_epoch >= len(env.requests) - 1:
        return [0] * env.num_base_stations

    horizon_steps = max(1, int(horizon_steps))
    last_step_index = min(len(env.requests) - 1, env.cur_epoch + horizon_steps)
    if last_step_index <= env.cur_epoch:
        return [0] * env.num_base_stations

    allowed_per_bs = _allowed_contents_per_bs_at(env, env.cur_epoch)
    caches_before = [list(c) for c in env.bs_caches]
    user_bs_conns = env.user_bs_connections

    future_reqs_window: List[List[int]] = []
    for t in range(env.cur_epoch + 1, last_step_index + 1):
        future_reqs_window.append(env.requests[t])

    actions_per_bs: List[List[int]] = []
    for bs in range(env.num_base_stations):
        actions_per_bs.append(_enumerate_actions_for_bs(caches_before[bs], allowed_per_bs[bs], env.num_contents))

    total_combos = 1
    for a in actions_per_bs:
        total_combos *= max(1, len(a))
    if total_combos > MAX_JOINT_COMBOS_PER_STEP:
        _logger.warning(f"[exhaustive-H{horizon_steps}] 组合数 {total_combos} 超阈值，仍按枚举执行。")

    best_joint = [0] * env.num_base_stations
    best_avg = -1.0

    def _discounted_avg_for_caches(caches_after: List[List[int]]) -> float:
        scores = [
            _compute_instant_hit_rate(caches_after, reqs, user_bs_conns)
            for reqs in future_reqs_window
        ]
        if not scores:
            return 0.0
        weights = [DISCOUNT_GAMMA ** i for i in range(len(scores))]
        wsum = float(sum(weights)) or 1.0
        return float(sum(w * s for w, s in zip(weights, scores)) / wsum)

    if env.num_base_stations == 1:
        for a0 in actions_per_bs[0]:
            caches_after = [_apply_action_to_cache(caches_before[0], a0, env.num_contents)]
            avg_score = _discounted_avg_for_caches(caches_after)
            if avg_score > best_avg:
                best_avg, best_joint = avg_score, [a0]
        return best_joint

    for a0 in actions_per_bs[0]:
        cache0_after = _apply_action_to_cache(caches_before[0], a0, env.num_contents)
        for a1 in actions_per_bs[1]:
            cache1_after = _apply_action_to_cache(caches_before[1], a1, env.num_contents)
            avg_score = _discounted_avg_for_caches([cache0_after, cache1_after])
            if avg_score > best_avg:
                best_avg = avg_score
                best_joint = [a0, a1]
    return best_joint

# =========================
# SAC baseline：checkpoint loader + 离散动作适配
# =========================

class _SACGaussianPolicy(nn.Module):
    """两层 MLP 的 Gaussian policy（与 `sac_baseline_B5/*.pt` 中的 key 对齐）。"""

    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return mean, log_std


def _continuous_scalar_to_discrete_index(
    a: float,
    action_space_size: int,
    valid_mask: Optional[np.ndarray],
) -> int:
    """把 SAC 输出的标量 a∈[-1,1] 映射到离散动作 index，并投影到最近的 valid action。"""
    if action_space_size <= 1:
        return 0
    a = float(np.clip(a, -1.0, 1.0))
    idx_f = (a + 1.0) * 0.5 * float(action_space_size - 1)
    idx = int(round(idx_f))
    idx = max(0, min(action_space_size - 1, idx))

    if valid_mask is None:
        return idx
    if 0 <= idx < len(valid_mask) and bool(valid_mask[idx]):
        return idx

    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return 0
    nearest = int(valid_indices[int(np.argmin(np.abs(valid_indices - idx)))])
    return nearest


def _build_topk_freq_state(env: UnifiedMultiBSCacheEnv, bs_idx: int, k: int) -> np.ndarray:
    """将频率 deque 压缩为固定长度 3*k 的向量：每个窗口取 top-k 的频率计数（降序），再整体归一化到 [0,1]。"""
    k = int(k)
    if k <= 0:
        return np.zeros((0,), dtype=np.float32)

    def _topk_counts(deq) -> np.ndarray:
        from collections import Counter

        c = Counter()
        for step_set in deq:
            if isinstance(step_set, (set, frozenset, list, tuple)):
                for cid in step_set:
                    if isinstance(cid, (int, np.integer)):
                        v = int(cid)
                        if 0 <= v < env.num_contents:
                            c[v] += 1
            elif isinstance(step_set, (int, np.integer)):
                v = int(step_set)
                if 0 <= v < env.num_contents:
                    c[v] += 1
        top = [cnt for _, cnt in c.most_common(k)]
        if len(top) < k:
            top.extend([0] * (k - len(top)))
        return np.asarray(top, dtype=np.float32)

    try:
        freq_deques = env.bs_resource_freqs[bs_idx]
        short = _topk_counts(freq_deques.get("short", []))
        mid = _topk_counts(freq_deques.get("mid", []))
        long = _topk_counts(freq_deques.get("long", []))
    except Exception:
        short = np.zeros((k,), dtype=np.float32)
        mid = np.zeros((k,), dtype=np.float32)
        long = np.zeros((k,), dtype=np.float32)

    feat = np.concatenate([short, mid, long], axis=0).astype(np.float32)
    mx = float(np.max(feat)) if feat.size else 0.0
    if mx > 1e-6:
        feat = feat / mx
    return feat


class SACCheckpointAgent:
    """
    加载 SAC checkpoint（仅用 policy）并输出离散联合动作：
    - 若 policy 输入维度 == env.n_features_per_agent_obs：直接用 env 的 individual_observations[bs]
    - 否则若 policy 输入维度可被 3 整除：用 top-k 频率压缩特征（k = state_dim//3）
    - 动作：policy 输出标量 a∈[-1,1]，映射到离散动作 index，并投影到 action_mask 的最近合法动作
    """

    def __init__(self, policy: _SACGaussianPolicy, state_dim: int, action_dim: int, device: str):
        self.policy = policy.to(device)
        self.policy.eval()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device

    @torch.no_grad()
    def _act(self, state_vec: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).view(1, -1)
        mean, _ = self.policy(x)
        a = torch.tanh(mean)
        return a.squeeze(0).detach().cpu().numpy()

    def _build_state(self, env: UnifiedMultiBSCacheEnv, bs_idx: int, individual_observations: List[np.ndarray]) -> np.ndarray:
        if (0 <= bs_idx < len(individual_observations)) and (individual_observations[bs_idx].shape[0] == self.state_dim):
            return individual_observations[bs_idx].astype(np.float32, copy=False)
        if self.state_dim % 3 == 0:
            k = self.state_dim // 3
            return _build_topk_freq_state(env, bs_idx, k)
        raise ValueError(
            f"SAC policy state_dim={self.state_dim} 与环境观测不匹配："
            f"per_bs_obs_dim={getattr(env, 'n_features_per_agent_obs', None)}，且 state_dim 不能被3整除。"
        )

    def choose_joint_action(self, env, global_state, individual_observations, action_masks) -> List[int]:
        # 兜底 action_masks
        if action_masks is None or len(action_masks) != env.num_base_stations:
            _, _, action_masks = env._get_obs_and_mask()

        joint_actions: List[int] = []
        for bs in range(env.num_base_stations):
            state_vec = self._build_state(env, bs, individual_observations)
            a_vec = self._act(state_vec)
            if a_vec.size == 0:
                joint_actions.append(0)
                continue
            # 目前默认每 BS 一个标量动作
            a0 = float(a_vec[0])
            space_size = int(env.action_spaces_per_bs[bs])
            mask = action_masks[bs] if bs < len(action_masks) else None
            joint_actions.append(_continuous_scalar_to_discrete_index(a0, space_size, mask))
        return joint_actions


def load_sac_checkpoint_agent(ckpt_path: str) -> Optional[SACCheckpointAgent]:
    if not ckpt_path:
        return None
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isdir(ckpt_path):
        # 兼容用户传入 checkpoint 目录：优先 sac_final.pt，否则挑一个最“像最新”的 .pt
        default_pt = os.path.join(ckpt_path, "sac_final.pt")
        if os.path.isfile(default_pt):
            ckpt_path = default_pt
        else:
            pt_files = sorted(glob.glob(os.path.join(ckpt_path, "*.pt")))
            if not pt_files:
                _logger.warning(f"SAC checkpoint 目录下未找到 .pt：{ckpt_path}")
                return None

            def _step_num(p: str) -> int:
                bn = os.path.basename(p)
                m = re.search(r"(?:sac_step_|step_)(\\d+)", bn)
                if m:
                    return int(m.group(1))
                m = re.search(r"(\\d+)", bn)
                return int(m.group(1)) if m else -1

            ckpt_path = max(pt_files, key=lambda p: (_step_num(p), os.path.getmtime(p)))
            _logger.info(f"SAC checkpoint 输入为目录，自动选择：{ckpt_path}")
    if not os.path.isfile(ckpt_path):
        _logger.warning(f"SAC checkpoint 不存在：{ckpt_path}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        _logger.error(f"加载 SAC checkpoint 失败：{e}")
        return None

    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        policy_sd = ckpt["policy_state_dict"]
    elif isinstance(ckpt, dict):
        policy_sd = ckpt
    else:
        _logger.error("SAC checkpoint 格式不支持（期望 dict 或包含 policy_state_dict）。")
        return None

    try:
        state_dim = int(policy_sd["linear1.weight"].shape[1])
        hidden_dim = int(policy_sd["linear1.weight"].shape[0])
        action_dim = int(policy_sd["mean_linear.weight"].shape[0])
    except Exception as e:
        _logger.error(f"无法从 policy_state_dict 推断维度：{e}")
        return None

    policy = _SACGaussianPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    try:
        policy.load_state_dict(policy_sd, strict=True)
    except Exception:
        policy.load_state_dict(policy_sd, strict=False)

    _logger.info(
        f"SAC checkpoint 加载成功：{ckpt_path} | state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}"
    )
    return SACCheckpointAgent(policy=policy, state_dim=state_dim, action_dim=action_dim, device=device)

# =========================
# 数据冻结工具
# =========================
def _freeze_env_materials(data_loader: MultiUserDataLoaderZipf) -> Dict[str, Any]:
    """构建一个参考 env，抽取并冻结 requests & user_bs_connections 等材料。"""
    ref_env = UnifiedMultiBSCacheEnv(
        dataloader=data_loader,
        num_base_stations=NUM_BASE_STATIONS,
        cache_sizes=CACHE_SIZES,
        num_users=NUM_USERS,
        num_contents=NUM_CONTENTS
    )
    frozen = {
        "requests": copy.deepcopy(ref_env.requests),
        "user_bs_connections": copy.deepcopy(ref_env.user_bs_connections),
        "num_epochs": ref_env.num_epochs,
    }
    # 给 requests 做个签名，方便日志里对比不同策略是否一致
    arr = np.array(frozen["requests"], dtype=np.int32)
    digest = hashlib.md5(arr.tobytes()).hexdigest()
    frozen["signature_md5"] = digest
    _logger.info(f"[数据冻结] requests MD5 = {digest} | epochs={ref_env.num_epochs}")
    return frozen

def _apply_frozen_materials(env: UnifiedMultiBSCacheEnv, frozen: Dict[str, Any]) -> None:
    """把冻结的数据材料灌入新 env。"""
    env.requests = copy.deepcopy(frozen["requests"])
    env.user_bs_connections = copy.deepcopy(frozen["user_bs_connections"])
    env.num_epochs = frozen.get("num_epochs", env.num_epochs)

# =========================
# 评估主流程
# =========================
def run_evaluation(
    agent_type: str,
    start_points: List[int],
    data_loader: MultiUserDataLoaderZipf,
    cache_validator: CacheActionValidator,
    model: Any = None,
    tokenizer: Optional[AutoTokenizer] = None,
    decode_mode: str = "greedy",  # greedy | sample_pass4
    adapter_name: Optional[str] = None,             # 进入评估前设置 adapter
    frozen_data: Optional[Dict[str, Any]] = None,   # 确保所有策略用同一份数据
    rng_seed: Optional[int] = None,                 # 评估用的随机种子
) -> Dict[str, Any]:
    # 使用传入种子；若未指定则回落到 TRAINER_SEED
    seed_to_use = TRAINER_SEED if rng_seed is None else int(rng_seed)
    torch.manual_seed(seed_to_use)
    np.random.seed(seed_to_use)
    random.seed(seed_to_use)
    set_seed(seed_to_use)

    _logger.info(f"\n{'='*40} 评估: {agent_type.upper()} （解码模式: {decode_mode}）{'='*40}")
    if frozen_data is not None:
        _logger.info(f"[评估数据签名] requests MD5 = {frozen_data.get('signature_md5')}, adapter={adapter_name}")

    results = {"agent_type": agent_type, "evaluations": []}
    all_hit_rates: List[float] = []
    decision_times_ms: List[float] = []
    prefix_per_episode: List[Dict[str, float]] = []

    # 若提供 adapter_name，则切换
    if adapter_name and model is not None and hasattr(model, "set_adapter"):
        try:
            model.set_adapter(adapter_name)
            _logger.info(f"已切换激活 LoRA 适配器为: {adapter_name}")
        except Exception as e:
            _logger.warning(f"切换适配器 {adapter_name} 失败：{e}")

    eos_ids = None
    if tokenizer is not None:
        try:
            eos_set = {
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            }
            eos_ids = [t for t in eos_set if isinstance(t, int) and t >= 0]
        except Exception:
            eos_ids = [tokenizer.eos_token_id] if getattr(tokenizer, "eos_token_id", None) is not None else None

    for initial_epoch_id in tqdm(start_points, desc=f"评估 {agent_type}"):
        # 新 env
        env = UnifiedMultiBSCacheEnv(
            dataloader=data_loader,
            num_base_stations=NUM_BASE_STATIONS,
            cache_sizes=CACHE_SIZES,
            num_users=NUM_USERS,
            num_contents=NUM_CONTENTS
        )
        # 灌入冻结数据，保证所有策略一致
        if frozen_data is not None:
            _apply_frozen_materials(env, frozen_data)

        # 不要 reset_freqs（保持 requests 不变）
        _initialize_env_for_sample(env, initial_epoch_id, cache_validator, reset_freqs=False)

        if agent_type == "sac" and model is not None:
            global_state, individual_observations, action_masks = env._get_obs_and_mask()

        episode_log = {"initial_epoch_id": initial_epoch_id, "hit_rates": []}

        for step in range(NUM_SIMULATION_STEPS):
            if env.cur_epoch >= len(env.requests) - 2:
                _logger.warning("达到数据末尾，提前结束评估。")
                break

            joint_actions: List[int] = []

            if agent_type in ('grpo_five_llm', 'grpo_ten_llm', 'sft_llm'):
                requested_sets = env._get_requested_contents_per_bs(env.requests[env.cur_epoch])
                messages = format_joint_state_to_llm_prompt(env, requested_sets)

                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([prompt], return_tensors="pt")
                try:
                    inputs = inputs.to(model.device)
                except Exception:
                    pass

                if decode_mode == 'greedy':
                    gen_cfg = _build_greedy_gen_config(tokenizer, eos_ids)
                    n_passes = 1
                elif decode_mode == 'sample_pass4':
                    gen_cfg = _build_sample_gen_config(tokenizer, eos_ids)
                    n_passes = 4
                else:
                    _logger.warning(f"未知 decode_mode={decode_mode} ，回退为 greedy")
                    gen_cfg = _build_greedy_gen_config(tokenizer, eos_ids)
                    n_passes = 1

                # 仅统计生成时间（不含解析/评分）
                t0 = time.perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        generation_config=gen_cfg,
                        num_return_sequences=n_passes,
                    )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                decision_times_ms.append(float(elapsed_ms))

                # 统一为列表
                decoded_list: List[str] = []
                if output_ids is not None:
                    decoded_list = tokenizer.batch_decode(
                        output_ids[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    )
                    if n_passes == 1 and decoded_list:
                        decoded_list = [decoded_list[0]]
                else:
                    decoded_list = []

                # 解析多样本，择优：按"下一步即时命中率"选择最佳
                candidates: List[List[int]] = []
                for text_out in decoded_list:
                    parsed_actions = cache_validator.parse_to_actions_or_none(text_out.strip())
                    if parsed_actions:
                        ja = convert_parsed_action_to_index(parsed_actions, NUM_CONTENTS, NUM_BASE_STATIONS)
                        ja = sanitize_joint_actions(env, ja)
                        candidates.append(ja)

                if not candidates:
                    _logger.warning("LLM输出（含采样）均解析失败或未通过校验，使用 no-op。")
                    joint_actions = [0] * NUM_BASE_STATIONS
                else:
                    # 去重并选择分数最高的候选
                    uniq = []
                    seen = set()
                    for ja in candidates:
                        tup = tuple(ja)
                        if tup not in seen:
                            uniq.append(ja)
                            seen.add(tup)
                    best_score = -1.0
                    best_ja = uniq[0]
                    for ja in uniq:
                        s = _score_joint_action_single_step(env, ja)
                        if s > best_score:
                            best_score = s
                            best_ja = ja
                    joint_actions = best_ja

            elif agent_type == "sac":
                if model is None:
                    _logger.warning("SAC 模型未加载，跳过该回合。")
                    break
                t0 = time.perf_counter()
                joint_actions = model.choose_joint_action(env, global_state, individual_observations, action_masks)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                decision_times_ms.append(float(elapsed_ms))

            elif agent_type == 'exhaustive':
                t0 = time.perf_counter()
                joint_actions = choose_exhaustive_joint_action_single_step(env)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                decision_times_ms.append(float(elapsed_ms))

            elif agent_type == 'five_step_tail_noop':
                t0 = time.perf_counter()
                joint_actions = choose_exhaustive_joint_action_horizon_noop(env, 5)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                decision_times_ms.append(float(elapsed_ms))

            elif agent_type == 'three_step_tail_noop':
                t0 = time.perf_counter()
                joint_actions = choose_exhaustive_joint_action_horizon_noop(env, 3)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                decision_times_ms.append(float(elapsed_ms))

            else:  # 传统策略（LRU / LFU / FIFO）
                t0 = time.perf_counter()
                requested_sets = env._get_requested_contents_per_bs(env.requests[env.cur_epoch])
                for bs_idx in range(NUM_BASE_STATIONS):
                    act = 0
                    candidates = sorted([c for c in requested_sets[bs_idx] if c not in env.bs_caches[bs_idx]])
                    if candidates:
                        content_to_cache = candidates[0]
                        slot_to_replace = -1
                        try:
                            slot_to_replace = env.bs_caches[bs_idx].index(-1)
                        except ValueError:
                            if agent_type == 'lru':
                                slot_to_replace = int(np.argmin(env.bs_used_times[bs_idx]))
                            elif agent_type == 'lfu':
                                slot_to_replace = int(np.argmin(env.bs_access_counts[bs_idx]))
                            elif agent_type == 'fifo':
                                slot_to_replace = int(np.argmin(env.bs_cached_times[bs_idx]))
                        if 0 <= slot_to_replace < len(env.bs_caches[bs_idx]):
                            act = slot_to_replace * NUM_CONTENTS + content_to_cache + 1
                    joint_actions.append(act)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                decision_times_ms.append(float(elapsed_ms))

            _, _, _, done, info = env.step(joint_actions)
            hit_rate = float(info.get('instant_hit_rate', 0.0))
            episode_log["hit_rates"].append(hit_rate)
            all_hit_rates.append(hit_rate)

            if agent_type == "sac" and model is not None:
                global_state, individual_observations, action_masks = env._get_obs_and_mask()

            if done:
                break

        prefix_avg = _compute_prefix_avg(episode_log["hit_rates"], PREFIX_AVG_STEPS)
        episode_log["prefix_avg_hit_rate"] = prefix_avg
        prefix_per_episode.append(prefix_avg)
        results["evaluations"].append(episode_log)

    results["overall_avg_hit_rate"] = float(np.mean(all_hit_rates)) if all_hit_rates else 0.0
    # 计算前缀平均（跨多个起点取均值/标准差）
    prefix_mean: Dict[str, float] = {}
    prefix_std: Dict[str, float] = {}
    for t in PREFIX_AVG_STEPS:
        key = str(t)
        vals = [p[key] for p in prefix_per_episode if key in p]
        if vals:
            prefix_mean[key] = float(np.mean(vals))
            prefix_std[key] = float(np.std(vals, ddof=0))
    results["prefix_avg_hit_rate"] = prefix_mean
    results["prefix_avg_hit_rate_std"] = prefix_std
    results["avg_decision_time_per_step_ms"] = float(np.mean(decision_times_ms)) if decision_times_ms else 0.0
    results["num_decisions"] = int(len(decision_times_ms))
    _logger.info(
        f"{agent_type.upper()} 评估完成 | 平均命中率: {results['overall_avg_hit_rate']:.6f} | "
        f"平均决策时间: {results['avg_decision_time_per_step_ms']:.2f} ms/step"
    )
    return results

# =========================
# 主函数
# =========================
def main():
    global NUM_BASE_STATIONS, NUM_CONTENTS, CACHE_SIZES, ZIPF_PARAM, NUM_USERS_LIST_TO_TEST, OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="统一缓存策略评估（SFT/GRPO/DAPO LLM + SAC baseline + 启发式；LLM 默认 greedy，可选采样）"
    )
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="输出文件名")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="评估结果输出目录")
    parser.add_argument("--num_points", type=int, default=NUM_EVALUATION_START_POINTS, help="评估起点数量")
    parser.add_argument("--num_steps", type=int, default=NUM_SIMULATION_STEPS, help="每个评估的步数")
    parser.add_argument("--include_pass4", action="store_true", help="可选：加入采样评估（4 次采样择优）")
    parser.add_argument("--num_base_stations", type=int, default=NUM_BASE_STATIONS, help="基站数量（如 2 或 5）")
    parser.add_argument("--cache_sizes", type=str, default="", help="每个基站 cache 大小，用逗号分隔；为空则全部默认=10")
    parser.add_argument("--num_contents", type=int, default=NUM_CONTENTS, help="内容总数")
    parser.add_argument("--zipf_param", type=float, default=ZIPF_PARAM, help="Zipf 参数")
    parser.add_argument("--num_users_list", type=str, default=",".join(map(str, NUM_USERS_LIST_TO_TEST)), help="评估的用户数量列表（逗号分隔），如只测40则写 40")
    parser.add_argument("--grpo_five_lora_dir", type=str, default=GRPO_FIVE_LORA_DIR_DEFAULT, help="GRPO_FIVE LoRA 目录（包含 adapter）")
    parser.add_argument("--grpo_ten_lora_dir", type=str, default=GRPO_TEN_LORA_DIR_DEFAULT, help="GRPO_TEN LoRA 目录（包含 adapter）")
    parser.add_argument("--sac_ckpt", type=str, default=SAC_CKPT_DEFAULT, help="SAC checkpoint 路径（.pt）")
    parser.add_argument("--num_seeds", type=int, default=3, help="多次评估的种子数量并取平均（默认3）")
    parser.add_argument("--seed_base", type=int, default=GRPO_DATA_SEED, help="数据生成起始种子（将顺延 num_seeds 次）")
    args = parser.parse_args()

    # =======================================================
    # 用命令行覆盖全局配置（便于分别测试 2 基站 / 5 基站）
    # =======================================================
    def _parse_csv_int_list(s: str) -> List[int]:
        s = (s or "").strip()
        if not s:
            return []
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        return [int(p) for p in parts]

    NUM_BASE_STATIONS = int(args.num_base_stations)
    NUM_CONTENTS = int(args.num_contents)
    ZIPF_PARAM = float(args.zipf_param)
    OUTPUT_DIR = os.path.abspath(args.output_dir)

    parsed_cache_sizes = _parse_csv_int_list(args.cache_sizes)
    if not parsed_cache_sizes:
        CACHE_SIZES = [10] * NUM_BASE_STATIONS
    elif len(parsed_cache_sizes) == 1:
        CACHE_SIZES = [int(parsed_cache_sizes[0])] * NUM_BASE_STATIONS
    elif len(parsed_cache_sizes) == NUM_BASE_STATIONS:
        CACHE_SIZES = [int(x) for x in parsed_cache_sizes]
    else:
        raise ValueError(f"--cache_sizes 长度必须为 1 或等于 num_base_stations={NUM_BASE_STATIONS}，当前={parsed_cache_sizes}")

    parsed_users_list = _parse_csv_int_list(args.num_users_list)
    if not parsed_users_list:
        raise ValueError("--num_users_list 不能为空")
    NUM_USERS_LIST_TO_TEST = [int(x) for x in parsed_users_list]

    # 默认只评估 greedy；可选加入采样
    decode_modes = list(DECODE_CHOICES)
    if args.include_pass4 and "sample_pass4" not in decode_modes:
        decode_modes.append("sample_pass4")
    _logger.info(f"本次 LLM 解码模式集合：{decode_modes}")
    _logger.info(f"评估配置：BS={NUM_BASE_STATIONS}, cache_sizes={CACHE_SIZES}, contents={NUM_CONTENTS}, zipf={ZIPF_PARAM}, users_list={NUM_USERS_LIST_TO_TEST}")

    # ==== 随机种子（全局一次性）====
    torch.manual_seed(TRAINER_SEED)
    np.random.seed(TRAINER_SEED)
    random.seed(TRAINER_SEED)
    set_seed(TRAINER_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    original_output_file = args.output  # 保存原始输出文件名

    # ===== 加载 SFT 基座 & LoRA 适配器（一次加载，多种子复用，且与 NUM_USERS 无关）=====
    models: Dict[str, Dict[str, Any]] = {"grpo_five_llm": {}, "grpo_ten_llm": {}, "sft_llm": {}, "sac": {}}

    tokenizer = None
    sft_model_for_adapters = None  # 共享基座：装载多个 adapter 并切换
    sft_model_baseline = None      # 纯SFT基线

    if FastLanguageModel is not None and _TRANSFORMERS_AVAILABLE and os.path.isdir(MODEL_PATH):
        try:
            # 1) 纯 SFT 基线（独立实例，避免adapter状态污染）
            sft_model_baseline, tokenizer = FastLanguageModel.from_pretrained(
                MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True, device_map="auto",
                dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            )
            sft_model_baseline.eval()
            try:
                sft_model_baseline.config.use_cache = False
            except Exception:
                pass
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            tokenizer.padding_side = 'left'
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            _clean_model_generation_config(sft_model_baseline)
            models["sft_llm"] = {"model": sft_model_baseline, "tokenizer": tokenizer, "adapter_name": None}
            _logger.info("SFT 基座加载成功。")

            # 2) 共享基座用于 LoRA（同一实例加载两个 adapter）
            sft_model_for_adapters, _ = FastLanguageModel.from_pretrained(
                MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True, device_map="auto",
                dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            )
            sft_model_for_adapters.eval()
            try:
                sft_model_for_adapters.config.use_cache = False
            except Exception:
                pass
            _clean_model_generation_config(sft_model_for_adapters)

            # ===== GRPO_FIVE LoRA =====
            grpo_five_dir = args.grpo_five_lora_dir
            if os.path.isdir(grpo_five_dir):
                loaded = False
                try:
                    sft_model_for_adapters.load_adapter(grpo_five_dir, "grpo_five")
                    loaded = True
                except Exception:
                    try:
                        sft_model_for_adapters.load_adapter(grpo_five_dir)  # 默认名
                        _logger.warning("GRPO_FIVE 适配器未显式命名，使用默认名称。")
                        loaded = True
                    except Exception:
                        try:
                            from peft import PeftModel
                            sft_model_for_adapters = PeftModel.from_pretrained(sft_model_for_adapters, grpo_five_dir)
                            _logger.warning("GRPO_FIVE 通过 PEFT 兼容加载（可能覆盖多adapter能力）。")
                            loaded = True
                        except Exception as e_peft:
                            _logger.error(f"通过 PEFT 加载 GRPO_FIVE 适配器失败: {e_peft}")
                if loaded:
                    _clean_model_generation_config(sft_model_for_adapters)
                    models["grpo_five_llm"] = {"model": sft_model_for_adapters, "tokenizer": tokenizer, "adapter_name": "grpo_five"}
                    _logger.info("GRPO_FIVE LoRA 权重加载成功（共享基座）。")
                else:
                    _logger.error(f"无法从 {grpo_five_dir} 加载 GRPO_FIVE LoRA 适配器。")
            else:
                _logger.warning(f"GRPO_FIVE LoRA 目录不存在: {grpo_five_dir}")

            # ===== GRPO_TEN LoRA =====
            grpo_ten_dir = args.grpo_ten_lora_dir
            if os.path.isdir(grpo_ten_dir):
                loaded2 = False
                try:
                    sft_model_for_adapters.load_adapter(grpo_ten_dir, "grpo_ten")
                    loaded2 = True
                except Exception:
                    try:
                        sft_model_for_adapters.load_adapter(grpo_ten_dir)  # 默认名
                        _logger.warning("GRPO_TEN 适配器未显式命名，使用默认名称。")
                        loaded2 = True
                    except Exception:
                        try:
                            from peft import PeftModel
                            sft_model_for_adapters = PeftModel.from_pretrained(sft_model_for_adapters, grpo_ten_dir)
                            _logger.warning("GRPO_TEN 通过 PEFT 兼容加载（可能覆盖多adapter能力）。")
                            loaded2 = True
                        except Exception as e_peft:
                            _logger.error(f"通过 PEFT 加载 GRPO_TEN 适配器失败: {e_peft}")
                if loaded2:
                    _clean_model_generation_config(sft_model_for_adapters)
                    models["grpo_ten_llm"] = {"model": sft_model_for_adapters, "tokenizer": tokenizer, "adapter_name": "grpo_ten"}
                    _logger.info("GRPO_TEN LoRA 权重加载成功（共享基座）。")
                else:
                    _logger.error(f"无法从 {grpo_ten_dir} 加载 GRPO_TEN LoRA 适配器。")
            else:
                _logger.warning(f"GRPO_TEN LoRA 目录不存在: {grpo_ten_dir}")

        except Exception as e:
            _logger.error(f"加载 SFT/LoRA LLM 失败: {e}", exc_info=True)
    else:
        if FastLanguageModel is None:
            _logger.warning("未安装 unsloth，跳过 LLM（SFT/GRPO）评估，只评估 SAC/启发式。")
        elif not _TRANSFORMERS_AVAILABLE:
            _logger.warning("未安装 transformers，跳过 LLM（SFT/GRPO）评估，只评估 SAC/启发式。")
        else:
            _logger.error(f"SFT 基座路径不存在：{MODEL_PATH}，无法进行 LLM 评估。")

    # ===== 加载 SAC baseline checkpoint（与 NUM_USERS 无关，可复用）=====
    sac_agent = load_sac_checkpoint_agent(args.sac_ckpt)
    if sac_agent is not None:
        models["sac"] = {"model": sac_agent}
    else:
        _logger.warning("未能加载 SAC checkpoint，将跳过 SAC 评估。")

    # =======================================================
    # 开始 NUM_USERS 循环测试
    # =======================================================
    for current_num_users in NUM_USERS_LIST_TO_TEST:
        global NUM_USERS
        NUM_USERS = current_num_users
        _logger.info(f"\n{'#'*60}\n正在测试 NUM_USERS = {NUM_USERS}\n{'#'*60}")

        # 更新输出文件名以区分不同用户数量
        fname, fext = os.path.splitext(original_output_file)
        output_filepath = os.path.join(OUTPUT_DIR, f"{fname}_users{NUM_USERS}{fext}")

        _logger.info("初始化缓存验证器（对齐训练脚本的构造方式）...")
        cache_validator = CacheActionValidator(NUM_CONTENTS, CACHE_SIZES)

        # 评估起点：从可用 epoch 中均匀抽样（与种子无关，保持索引一致）
        all_possible_epochs = list(range(NUM_EPOCHS_FOR_GRPO_DATA - args.num_steps - FUTURE_STEPS_REWARD - 5))
        if not all_possible_epochs:
            _logger.critical("数据长度不足以进行评估！")
            sys.exit(1)
        random.seed(42)
        chosen_epochs = sorted(random.sample(all_possible_epochs, min(args.num_points, len(all_possible_epochs))))
        _logger.info(f"选择的评估起点 (Epochs): {chosen_epochs}")

        # 组装多种子列表
        seeds = [int(args.seed_base) + i for i in range(int(args.num_seeds))]
        _logger.info(f"本次评估使用的种子集合: {seeds}")

        # 清理显存（多用户数循环时可选）
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # ===== 逐策略评估（多种子 & 多解码模式）=====
        strategies = [
            ("grpo_five_llm", "GRPO_FIVE LLM"),
            ("grpo_ten_llm", "GRPO_TEN LLM"),
            ("sft_llm", "SFT LLM"),
            ("sac", "SAC"),
            ("lru", "LRU"),
            ("lfu", "LFU"),
            ("fifo", "FIFO"),
            ("exhaustive", "单步穷举"),
        ]
        if NUM_BASE_STATIONS <= 2:
            strategies.extend(
                [
                    ("five_step_tail_noop", "五步穷举(尾部NoOp,贴现)"),
                    ("three_step_tail_noop", "三步穷举(尾部NoOp,贴现)"),
                ]
            )

        per_seed_results = []  # 每个元素对应一个种子的一整套策略结果

        for seed in seeds:
            _logger.info(f"\n{'='*20} 开始 Seed = {seed} 的评估 (Users={NUM_USERS}) {'='*20}\n")
            # 为该种子构造数据加载器，并冻结材料（requests / 连接关系）
            data_loader = MultiUserDataLoaderZipf(
                NUM_CONTENTS, NUM_EPOCHS_FOR_GRPO_DATA, NUM_USERS, ZIPF_PARAM, seed=seed
            )
            frozen_data = _freeze_env_materials(data_loader)

            seed_results_this_round = []
            for strategy_id, strategy_name in strategies:
                model_obj = models.get(strategy_id, {})

                # LLM 策略：对每个 decode_mode 都评估一次
                if "llm" in strategy_id:
                    if not model_obj.get("model"):
                        _logger.warning(f"[seed={seed}] 跳过评估 {strategy_name} 因为模型未加载或路径缺失。")
                        continue
                    for dm in decode_modes:
                        try:
                            result = run_evaluation(
                                strategy_id,
                                chosen_epochs,
                                data_loader,
                                cache_validator,
                                decode_mode=dm,
                                frozen_data=frozen_data,
                                adapter_name=model_obj.get("adapter_name"),
                                model=model_obj.get("model"),
                                tokenizer=model_obj.get("tokenizer"),
                                rng_seed=seed,
                            )
                            result["strategy_name"] = f"{strategy_name} ({dm})"
                            result["seed"] = seed
                            seed_results_this_round.append(result)
                        except Exception as e:
                            _logger.error(f"[seed={seed}] {strategy_name} ({dm}) 评估失败: {e}", exc_info=True)
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                else:
                    # 非 LLM：只评估一次
                    if strategy_id == "sac" and (not model_obj.get("model")):
                        _logger.warning(f"[seed={seed}] 跳过评估 {strategy_name}：未成功加载 SAC checkpoint。")
                        continue
                    try:
                        result = run_evaluation(
                            strategy_id,
                            chosen_epochs,
                            data_loader,
                            cache_validator,
                            frozen_data=frozen_data,
                            adapter_name=model_obj.get("adapter_name"),
                            model=model_obj.get("model"),
                            tokenizer=model_obj.get("tokenizer"),
                            rng_seed=seed,
                        )
                        result["strategy_name"] = strategy_name
                        result["seed"] = seed
                        seed_results_this_round.append(result)
                    except Exception as e:
                        _logger.error(f"[seed={seed}] {strategy_name} 评估失败: {e}", exc_info=True)
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            per_seed_results.append({"seed": seed, "results": seed_results_this_round})

        # ===== 聚合：按策略计算 mean / std（overall_avg_hit_rate 与 avg_decision_time_per_step_ms）=====
        agg_rate_map: Dict[str, List[float]] = {}
        agg_time_map: Dict[str, List[float]] = {}
        agg_prefix_map: Dict[str, Dict[str, List[float]]] = {}

        for seed_pack in per_seed_results:
            for res in seed_pack["results"]:
                name = res.get("strategy_name", res.get("agent_type", "unknown"))
                rate = float(res.get("overall_avg_hit_rate", 0.0))
                tavg = float(res.get("avg_decision_time_per_step_ms", 0.0))
                agg_rate_map.setdefault(name, []).append(rate)
                agg_time_map.setdefault(name, []).append(tavg)
                prefix_map = res.get("prefix_avg_hit_rate", {})
                if isinstance(prefix_map, dict):
                    for step_str, val in prefix_map.items():
                        agg_prefix_map.setdefault(name, {}).setdefault(step_str, []).append(float(val))

        aggregate = []
        for name in sorted(set(list(agg_rate_map.keys()) + list(agg_time_map.keys()))):
            rates = agg_rate_map.get(name, [])
            times = agg_time_map.get(name, [])
            mean_v = float(np.mean(rates)) if rates else 0.0
            std_v  = float(np.std(rates, ddof=0)) if rates else 0.0
            mean_t = float(np.mean(times)) if times else 0.0
            std_t  = float(np.std(times, ddof=0)) if times else 0.0
            prefix_mean: Dict[str, float] = {}
            prefix_std: Dict[str, float] = {}
            step_map = agg_prefix_map.get(name, {})
            for step_str, vals in step_map.items():
                if vals:
                    prefix_mean[step_str] = float(np.mean(vals))
                    prefix_std[step_str] = float(np.std(vals, ddof=0))

            aggregate.append({
                "strategy_name": name,
                "mean_overall_avg_hit_rate": mean_v,
                "std_overall_avg_hit_rate": std_v,
                "mean_decision_time_ms": mean_t,
                "std_decision_time_ms": std_t,
                "prefix_avg_hit_rate_mean": convert_numpy_to_python(prefix_mean),
                "prefix_avg_hit_rate_std": convert_numpy_to_python(prefix_std),
                "rates": convert_numpy_to_python(rates),
                "avg_times_ms": convert_numpy_to_python(times),
            })

        # 保存结果（包含每个种子的详细结果 & 聚合）
        payload = {
            "seeds": seeds,
            "num_users": NUM_USERS,
            "num_points": args.num_points,
            "num_steps": args.num_steps,
            "per_seed": convert_numpy_to_python(per_seed_results),
            "aggregate": convert_numpy_to_python(aggregate),
        }
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
        _logger.info(f"多种子评估结果已保存到 {output_filepath}")

        # 汇总报告（命中率 & 决策时间）
        _logger.info("\n" + "=" * 70 + f"\n多种子最终评估汇总报告 (Users={NUM_USERS})\n" + "=" * 70)
        for item in sorted(aggregate, key=lambda x: x["mean_overall_avg_hit_rate"], reverse=True):
            _logger.info(
                f"{item['strategy_name']:>36}: "
                f"Hit {item['mean_overall_avg_hit_rate']:.6f} ± {item['std_overall_avg_hit_rate']:.6f} | "
                f"Time {item['mean_decision_time_ms']:.2f} ± {item['std_decision_time_ms']:.2f} ms/step"
            )
        _logger.info("=" * 70)

if __name__ == "__main__":
    main()

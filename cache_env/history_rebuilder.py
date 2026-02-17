# -*- coding: utf-8 -*-
# 文件: cache/history_rebuilder.py

from typing import Set, Dict, List, Tuple
from cache_env.unified_multi_bs_cache_env import UnifiedMultiBSCacheEnv

# =================================================================
# === 穷举-前瞻命中率教师（未来5步） ==================================
# =================================================================

LOOKAHEAD_HORIZON = 10  # 未来步数（固定为10步）
_EPS = 1e-9

# ---------------------- 工具：频率计数（按时间步集合） ----------------------
def _count_in_steps(step_deque, cid: int) -> int:
    """统计该 cid 出现在多少个“时间步集合”里。"""
    return sum(1 for step_set in step_deque if cid in step_set)

# ---------------------- 工具：动作编码/解码（与环境一致） -------------------
def _decode_action(env: UnifiedMultiBSCacheEnv, idx: int) -> Tuple[int, int]:
    """从动作索引解出 (slot, new_cid)。NO_OP 由上层判断。"""
    slot = (idx - 1) // env.num_contents
    cid  = (idx - 1) %  env.num_contents
    return slot, cid

def _encode_action(env: UnifiedMultiBSCacheEnv, slot: int, new_cid: int) -> int:
    """把 (slot, new_cid) 编码为动作索引。"""
    return int(slot * env.num_contents + new_cid + 1)

# ---------------------- 候选动作枚举 ---------------------------------------
def _enumerate_candidate_actions(
    env: UnifiedMultiBSCacheEnv,
    bs_idx: int,
    current_bs_requested_contents: Set[int],
) -> List[int]:
    """
    列举当前步对该基站可执行的动作（含不操作）：
    - 新内容必须来自该基站的当前请求；
    - 允许替换非空槽或向空槽填充；
    - 若请求都已在缓存中，则只有“不操作”。
    """
    actions = [env.NO_OP_ACTION_INDEX]
    cache = env.bs_caches[bs_idx]
    cached_set = set(int(c) for c in cache if c != -1)

    to_insert = [int(cid) for cid in current_bs_requested_contents if int(cid) not in cached_set]
    if not to_insert:
        return actions  # 无可引入内容 -> 只有不操作

    non_empty_slots = [i for i, c in enumerate(cache) if c != -1]
    empty_slots     = [i for i, c in enumerate(cache) if c == -1]

    # 非空槽替换
    for slot in non_empty_slots:
        for new_cid in to_insert:
            actions.append(_encode_action(env, slot, new_cid))

    # 空槽填充
    for slot in empty_slots:
        for new_cid in to_insert:
            actions.append(_encode_action(env, slot, new_cid))

    # 去重 & 排序（稳定性）
    actions = sorted(set(actions))
    return actions

# ---------------------- 评分：未来5步真实命中率（静态缓存） -------------------
def _score_action_by_lookahead_network_hit_rate(
    env: UnifiedMultiBSCacheEnv,
    caches_snapshot: List[List[int]],
    start_epoch: int,
    horizon: int,
) -> Tuple[int, int]:
    """
    在不再替换的前提下，从 start_epoch+1 开始滚动 horizon 步：
    - 逐用户统计真实命中（用户连接的任一基站缓存中包含其请求内容即命中）；
    - 累计未来总命中数与总请求数；
    返回 (total_hits, total_reqs)。
    """
    total_hits = 0
    total_reqs = 0

    for dt in range(1, horizon + 1):
        epoch = start_epoch + dt
        if epoch >= env.num_epochs:
            break

        epoch_requests = env.requests[epoch]
        for user_id, req_content in enumerate(epoch_requests):
            if req_content == -1:
                continue
            total_reqs += 1

            # 用户连接到的任一基站命中即视为命中
            hit = False
            if user_id < len(env.user_bs_connections):
                for bs_j in env.user_bs_connections[user_id]:
                    # 注意：caches_snapshot[bs_j] 是该时刻静态缓存
                    if req_content in caches_snapshot[bs_j]:
                        hit = True
                        break
            if hit:
                total_hits += 1

    return total_hits, total_reqs

# ---------------------- 平局规则：优先长期损失小 / 中期增益大 -----------------
def _tie_breaker_keys(
    env: UnifiedMultiBSCacheEnv,
    bs_idx: int,
    action_idx: int,
    old_cid: int,
    new_cid: int,
) -> Tuple[int, int, int, int]:
    """
    为并列时提供可解释的打破规则：
    - key1: -长期损失（越大越好，即长期损失越小越优先）
    - key2:  中期增益（越大越好）
    - key3:  -槽位索引（越大越好 -> 槽位索引越小越优先）
    - key4:  -新内容ID（越大越好 -> 内容ID越小越优先）
    """
    if action_idx == env.NO_OP_ACTION_INDEX:
        return (0, 0, 0, 0)

    freqs = env.bs_resource_freqs[bs_idx]
    def _trip(cid: int):
        if cid == -1: return (0, 0, 0)
        s = _count_in_steps(freqs['short'], cid)
        m = _count_in_steps(freqs['mid'],   cid)
        l = _count_in_steps(freqs['long'],  cid)
        return (s, m, l)

    slot, _ = _decode_action(env, action_idx)
    _, mid_new, long_new = _trip(new_cid)
    _, mid_old, long_old = _trip(old_cid) if old_cid != -1 else (0, 0, 0)

    long_loss = max(0, long_old - long_new)
    mid_gain  = max(0,  mid_new - mid_old)

    # 槽位/内容ID偏好：小槽位、小内容ID更优（便于输出稳定）
    return (-long_loss, mid_gain, -slot, -new_cid)

# ---------------------- 主函数：前瞻-穷举教师 ---------------------------------
def get_lookahead_oracle_decision(
    env: UnifiedMultiBSCacheEnv,
    bs_idx: int,
    current_bs_requested_contents: Set[int],
    horizon: int = LOOKAHEAD_HORIZON,
) -> int:
    """
    穷举所有可动作（含不操作）。对每个动作：
    1) 在当前缓存快照上仅执行一次该动作；
    2) 之后不再替换，滚动未来 horizon 步，按“真实用户连接”统计全网命中率；
    3) 选择累计命中率（总命中/总请求）最高的动作；并列用可解释规则打破。
    """
    # 候选动作
    actions = _enumerate_candidate_actions(env, bs_idx, current_bs_requested_contents)
    if not actions:
        return env.NO_OP_ACTION_INDEX

    # 当前缓存快照
    base_caches: List[List[int]] = [list(map(int, row)) for row in env.bs_caches]

    # 记录最佳
    best_action = env.NO_OP_ACTION_INDEX
    best_hits   = -1
    best_reqs   = 1  # 防止除零
    best_tie    = (0, 0, 0, 0)

    for act in actions:
        caches_copy = [row[:] for row in base_caches]

        # old/new cid 仅用于平局规则
        old_cid, new_cid = -1, -1
        if act != env.NO_OP_ACTION_INDEX:
            slot, new_cid = _decode_action(env, act)
            old_cid = caches_copy[bs_idx][slot]
            caches_copy[bs_idx][slot] = new_cid  # 只在当前步执行一次

        # 未来 horizon 步的累计命中/请求
        hits, reqs = _score_action_by_lookahead_network_hit_rate(
            env=env,
            caches_snapshot=caches_copy,
            start_epoch=env.cur_epoch,
            horizon=horizon,
        )

        # 比较：优先累计命中率（命中/请求）
        # 由于每步请求数可能不同，直接比较 hits*best_reqs 与 best_hits*reqs（交叉乘法）避免浮点误差
        better = (hits * best_reqs) > (best_hits * reqs)

        if not better and (hits * best_reqs) == (best_hits * reqs):
            # 平局：按长期损失小/中期增益大/小槽位/小内容ID
            tie_key = _tie_breaker_keys(env, bs_idx, act, old_cid, new_cid)
            better = tie_key > best_tie

        if better:
            best_action = act
            best_hits   = hits
            best_reqs   = reqs if reqs > 0 else 1
            best_tie    = _tie_breaker_keys(env, bs_idx, act, old_cid, new_cid)

    return best_action

# ---------------------- SFT 对外接口（替换为前瞻教师） -----------------------
def get_sft_teaching_decision(
    env: UnifiedMultiBSCacheEnv,
    bs_idx: int,
    current_bs_requested_contents: Set[int],
    cache_validator=None,
) -> int:
    """
    用“未来5步真实命中率”的穷举教师打标签。
    提示词/思维链仍只基于历史频率与当前请求，不泄露未来信息。
    """
    if not current_bs_requested_contents:
        return env.NO_OP_ACTION_INDEX
    return get_lookahead_oracle_decision(
        env=env,
        bs_idx=bs_idx,
        current_bs_requested_contents=current_bs_requested_contents,
        horizon=LOOKAHEAD_HORIZON,
    )

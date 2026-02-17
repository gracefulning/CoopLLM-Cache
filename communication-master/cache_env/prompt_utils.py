# -*- coding: utf-8 -*-
# ================== cache/prompt_utils.py ==================
from typing import Any, Dict, List, Set
from collections import Counter

def _get_full_freq_str(
    historical_requests,         # deque[frozenset[int]]
    relevant_cids: Set[int],
    extra_increment: Set[int] = None,   # 当前 t 步“去重 +1”的集合（仅用于展示）
) -> str:
    """
    为相关内容ID生成频率计数串。若提供 extra_increment，则对其中的每个 cid 额外 +1。
    输出顺序：频率降序 -> 内容ID升序。
    """
    if not relevant_cids:
        return "无"

    counts = Counter()
    # 按“时间步集合”累计：出现过就 +1
    for step_set in historical_requests:       # step_set: frozenset[int]
        for cid in step_set:
            counts[int(cid)] += 1

    # 展示端对 t 步再 +1（去重）
    if extra_increment:
        for cid in extra_increment:
            counts[int(cid)] += 1

    report_items = [(cid, counts.get(int(cid), 0)) for cid in relevant_cids]
    sorted_items = sorted(report_items, key=lambda item: (-item[1], item[0]))
    return "; ".join([f"内容{cid}({count}次)" for cid, count in sorted_items])


def format_joint_state_to_llm_prompt(
    env: Any,
    joint_requests_sets: List[Set[int]],
    current_requests_sets_at_t: List[Set[int]] = None,  # 传入 t 步请求（按 BS 去重）
) -> List[Dict[str, str]]:
    """
    将环境状态格式化为纯文本、对LLM友好的消息。
    注意：不提供任何决策规则或策略，只提供事实信息与输出格式/硬约束，留给模型自行学习。
    """
    # === 1) System Prompt（最小必要信息） ===
    cache_size = env.cache_sizes[0] if env.cache_sizes else '未知'
    task_description = (
        f"你是一位无线网络缓存策略专家。你的目标是最大化网络的整体缓存命中率。\n"
        f"每个基站拥有{cache_size}个缓存槽位，编号从1到{cache_size}。\n"
        f"你必须为全部{env.num_base_stations}个基站做出决策。"
    )

    format_rules = (
        "输出格式要求（必须严格遵守）：\n"
        f"- 直接输出共 {env.num_base_stations} 行，每行对应一个基站的最终决策，禁止任何额外内容或解释。\n"
        "- 允许的两种句式为：\n"
        "  • 基站X的决策是：用内容Y替换槽位Z中的内容W。\n"
        "  • 基站X的决策是：不执行任何缓存操作。"
    )

    constraints = (
        "硬性约束（必须满足）：\n"
        "A. 替换时，新内容Y必须来自该基站的当前请求列表。\n"
        "B. 若输出“用内容Y替换槽位Z中的内容W”，则W必须与状态信息中“槽位Z: 内容W”完全一致。\n"
        "C. 不允许用当前缓存中已存在的内容作为新内容去替换任何槽位。"
    )

    system_prompt = f"{task_description}\n\n{format_rules}\n\n{constraints}\n\n请基于下方提供的状态信息自行做出最优决策。"

    # === 2) User Prompt（仅呈现客观状态） ===
    user_prompt = (
        f"网络状态信息（时间步: {env.cur_epoch}）。\n"
    )

    for bs_idx in range(env.num_base_stations):
        user_prompt += f"\n--- 基站 {bs_idx} ---\n"
        cache = env.bs_caches[bs_idx]
        requests_set = joint_requests_sets[bs_idx]

        # 当前缓存
        cache_details = []
        for i, c_id in enumerate(cache):
            if c_id != -1:
                cache_details.append(f"  - 槽位{i+1}: 内容{int(c_id)}")
        cache_str = "\n".join(cache_details) if cache_details else "  (空)"
        user_prompt += f"当前缓存:\n{cache_str}\n"

        # 当前请求（按 BS 去重后排序）
        requests_list_sorted = sorted(list(map(int, requests_set)))
        user_prompt += f"当前请求: {requests_list_sorted or '无'}\n"

        # 相关内容集合：缓存 ∪ 请求
        relevant_cids = set(int(c) for c in cache if c != -1) | set(int(c) for c in requests_set)

        # t 步“去重 +1”的集合（若未显式传入，则用 joint_requests_sets）
        if current_requests_sets_at_t is not None:
            t_extra = set(int(x) for x in current_requests_sets_at_t[bs_idx])
        else:
            t_extra = set(int(x) for x in joint_requests_sets[bs_idx])

        # 短/中/长频率
        user_prompt += "相关内容历史热点:\n"
        if not relevant_cids:
            user_prompt += "  - 短期 (近10步): 无\n"
            user_prompt += "  - 中期 (近100步): 无\n"
            user_prompt += "  - 长期 (近1000步): 无\n"
        else:
            freq_deques = env.bs_resource_freqs[bs_idx]  # deque[frozenset]
            short_str = _get_full_freq_str(freq_deques['short'], relevant_cids, extra_increment=t_extra)
            mid_str   = _get_full_freq_str(freq_deques['mid'],   relevant_cids, extra_increment=t_extra)
            long_str  = _get_full_freq_str(freq_deques['long'],  relevant_cids, extra_increment=t_extra)
            user_prompt += f"  - 短期 (近10步): {short_str}\n"
            user_prompt += f"  - 中期 (近100步): {mid_str}\n"
            user_prompt += f"  - 长期 (近1000步): {long_str}\n"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# -*- coding: utf-8 -*-
# 文件: cache_action_format.py
# 描述: 严格的缓存动作验证器（仅支持“替换/无操作”，不支持“填充”）
# 说明: 仅接受以下两种严格格式（每个基站恰好一行）：
#   1) 基站<id>的决策是：用内容<new_cid>替换槽位<slot_1b>中的内容<old_cid>。
#   2) 基站<id>的决策是：不执行任何缓存操作。

import re
from typing import Dict, Any, List, Optional, Set

class CacheActionValidator:
    """
    解析与验证从 LLM 输出中提取的多个基站决策。
    - 仅支持两类动作：CacheReplace / CacheNoOp
    - 严格：每个基站恰好一行；参数范围有效；基站ID不重复；覆盖全部BS
    - 健壮：容错“中的的内容”、繁简混用、全角/半角冒号、少量空白差异
    - 槽位文本为 1 基（槽位1..N），内部统一转换为 0 基用于校验与落地
    """
    def __init__(self, num_contents: int, cache_sizes: List[int]):
        self.num_contents = num_contents
        self.cache_sizes = cache_sizes
        self.num_base_stations = len(cache_sizes)
        self.NO_OP_ACTION_INDEX = 0

        # 数字捕获（允许少量空格）
        num = r"\s*(\d+)\s*"
        # 容错“中的的内容” -> 使用可选的“的”
        中的内容 = r"中\s*的?\s*内容"

        # 严格格式：不允许任何前缀（如“缓存替换:”或“缓存无操作:”）
        # 兼容中文/全角冒号（通过 normalize 统一为半角）
        self.REPLACE_PATTERN = re.compile(
            rf"基站{num}的决策是\s*[:：]\s*用内容{num}替换槽位{num}{中的内容}{num}\s*[.。]?$"
        )
        # 无操作：允许“缓存”二字可选（不执行任何操作 / 不执行任何缓存操作）
        self.NOOP_PATTERN = re.compile(
            rf"基站{num}的决策是\s*[:：]\s*不执行任何(?:缓存|緩存)?操作\s*[.。]?$"
        )

    def _normalize(self, text: str) -> str:
        """统一预处理：繁简关键字、全/半角冒号、重复'的'、空白。"""
        s = (text or "").strip()
        # 统一繁简关键字
        s = s.replace("緩存", "缓存").replace("內容", "内容").replace("將", "将")
        # 统一冒号为半角
        s = s.replace("：", ":")
        # 去掉每行首尾多余空白
        s = "\n".join(line.strip() for line in s.splitlines())
        # 容错“中的的内容” -> “中的内容”
        s = re.sub(r"中\s*的\s*的\s*内容", "中的内容", s)
        # 统一“中的 内容”间隔
        s = re.sub(r"中\s*的?\s*内容", "中的内容", s)
        return s

    def parse_to_actions_or_none(self, llm_output: str) -> Optional[List[Dict[str, Any]]]:
        """
        严格解析：只有当每个基站都提供了格式正确且参数有效的决策时，才返回动作列表。
        否则返回 None。
        返回结构（按 station_id 排序的列表）：
        [
          {"action": "CacheReplace", "parameters": {"station_id": 0, "content_id": 138, "slot_index": 6, "content_to_evict_id": 61}},
          {"action": "CacheNoOp",    "parameters": {"station_id": 1}}
        ]
        """
        llm_output = self._normalize(llm_output)
        lines = [line for line in llm_output.splitlines() if line.strip()]

        # 行数必须严格等于基站数
        if len(lines) != self.num_base_stations:
            return None

        actions: List[Optional[Dict[str, Any]]] = [None] * self.num_base_stations
        seen_bs_indices = set()

        for line in lines:
            parsed_action: Optional[Dict[str, Any]] = None
            m_rep = self.REPLACE_PATTERN.fullmatch(line)
            m_nop = self.NOOP_PATTERN.fullmatch(line)

            try:
                if m_rep:
                    bs_idx_s, new_cid_s, slot_1b_s, old_cid_s = m_rep.groups()
                    bs_idx   = int(bs_idx_s)
                    new_cid  = int(new_cid_s)
                    slot_1b  = int(slot_1b_s)
                    old_cid  = int(old_cid_s)
                    slot_0b  = slot_1b - 1  # 文本1基 -> 内部0基

                    # 范围检查
                    if not (0 <= bs_idx < self.num_base_stations and
                            0 <= new_cid < self.num_contents and
                            0 <= old_cid < self.num_contents and
                            0 <= slot_0b < self.cache_sizes[bs_idx]):
                        return None

                    parsed_action = {
                        "action": "CacheReplace",
                        "parameters": {
                            "station_id": bs_idx,
                            "content_id": new_cid,
                            "slot_index": slot_0b,           # 0 基
                            "content_to_evict_id": old_cid,
                        }
                    }

                elif m_nop:
                    (bs_idx_s,) = m_nop.groups()
                    bs_idx = int(bs_idx_s)
                    if not (0 <= bs_idx < self.num_base_stations):
                        return None
                    parsed_action = {
                        "action": "CacheNoOp",
                        "parameters": {"station_id": bs_idx}
                    }

                else:
                    # 未匹配任何模式（包括含有前缀的情况）
                    return None

                # 基站ID不能重复，且必须覆盖所有BS
                cur_bs = parsed_action["parameters"]["station_id"]
                if cur_bs in seen_bs_indices:
                    return None
                seen_bs_indices.add(cur_bs)
                actions[cur_bs] = parsed_action

            except (ValueError, TypeError, IndexError):
                return None

        if len(seen_bs_indices) != self.num_base_stations or any(a is None for a in actions):
            return None

        return actions

    # === 新增：解析 + 强校验（与环境/请求一致性） ===========================
    def parse_and_validate(
        self,
        llm_output: str,
        env,
        requested_sets_per_bs: List[Set[int]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        在 parse 的基础上，结合环境与“当前请求集”做三条硬性校验：
        1) W 一致性：env.bs_caches[bs][slot] == old_cid
        2) 新内容来自当前请求：new_cid ∈ requested_sets_per_bs[bs]
        3) 新内容不在当前缓存：new_cid ∉ env.bs_caches[bs]
        全部通过才返回动作列表，否则返回 None。
        """
        actions = self.parse_to_actions_or_none(llm_output)
        if actions is None:
            return None

        try:
            for a in actions:
                bs = a["parameters"]["station_id"]
                if a["action"] == "CacheReplace":
                    slot = a["parameters"]["slot_index"]
                    new_cid = a["parameters"]["content_id"]
                    old_cid = a["parameters"]["content_to_evict_id"]

                    # 1) W 一致性
                    if env.bs_caches[bs][slot] != old_cid:
                        return None
                    # 2) 新内容来自当前请求
                    if new_cid not in requested_sets_per_bs[bs]:
                        return None
                    # 3) 新内容不能已在缓存
                    if new_cid in env.bs_caches[bs]:
                        return None
                elif a["action"] == "CacheNoOp":
                    pass
                else:
                    return None
        except Exception:
            return None

        return actions

    def get_action_details_from_index(self, action_index: int, bs_idx: int) -> Dict[str, Any]:
        """
        用于 SFT 数据生成：把离散动作索引还原为动作名称与参数（仅替换/无操作）。
        兼容字段：
        - "action": "CacheNoOp" | "CacheReplace"
        - "name"  : 同上（为兼容既有代码）
        """
        if action_index == self.NO_OP_ACTION_INDEX:
            return {"action": "CacheNoOp", "name": "CacheNoOp", "parameters": {}}
        base_index = action_index - 1
        slot_index = base_index // self.num_contents
        content_id = base_index % self.num_contents
        return {
            "action": "CacheReplace",
            "name": "CacheReplace",
            "parameters": {"slot_index": int(slot_index), "content_id": int(content_id)}
        }

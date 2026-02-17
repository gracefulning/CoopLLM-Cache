# -*- coding: utf-8 -*-
# ================== unified_multi_bs_cache_env.py ==================
import math
import random
import logging
from typing import List, Tuple, Dict, Set, Any, Callable
from collections import deque, defaultdict

import numpy as np

# 这是一个前向声明，用于类型提示，以避免循环导入
class CacheActionValidator:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

FEAT_TERMS = [10, 100, 1000]  # 短/中/长窗口，用于频率特征（单位：时间步）

def euclidean_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def dbm_to_linear(dbm_power: float) -> float:
    return 10 ** ((dbm_power - 30.0) / 10.0)

class UnifiedMultiBSCacheEnv:
    """统一的多基站缓存环境。"""
    def __init__(
        self,
        dataloader,
        num_base_stations: int = 2,
        cache_sizes: List[int] = None,
        num_users: int = 15,
        num_contents: int = 100,
        area_size: Tuple[float, float] = (8, 4),
        cell_radius_km: float = 2.2,
        bs_tx_power_dbm: float = 16.9,
        cloud_tx_power_dbm: float = 20.0,
        content_file_size_bits: int = 96,
        bandwidth_hz: float = 1e6,
        noise_power_spectral_density: float = -174.0,
        reward_type: str = "hit_rate",
        normalize: bool = True,
        reward_smoothing_alpha: float = 0.5,
        fixed_bs_locations: List[Tuple[float, float]] = None,
        fixed_user_locations: List[Tuple[float, float]] = None,
    ):
        _logger.info("正在初始化 UnifiedMultiBSCacheEnv...")

        self.requests = dataloader.get_requests()
        self.operations = dataloader.get_operations()
        self.num_epochs = len(self.requests)

        self.num_users = num_users
        self.num_base_stations = num_base_stations
        self.cache_sizes = cache_sizes if cache_sizes is not None else [20] * num_base_stations
        self.num_contents = num_contents
        self.area_size = area_size
        self.cell_radius_km = cell_radius_km

        self.bs_tx_power_linear = dbm_to_linear(bs_tx_power_dbm)
        self.cloud_tx_power_linear = dbm_to_linear(cloud_tx_power_dbm)
        self.content_file_size_bits = content_file_size_bits
        self.bandwidth_hz = bandwidth_hz
        self.noise_power_spectral_density_linear = dbm_to_linear(noise_power_spectral_density)

        self.T0 = 1e-3
        self.reward_type = reward_type
        self.normalize = normalize
        self.reward_smoothing_alpha = reward_smoothing_alpha

        self.bs_locations = fixed_bs_locations if fixed_bs_locations is not None else self._generate_bs_locations(num_base_stations, self.area_size)
        self.user_locations = fixed_user_locations if fixed_user_locations is not None else self._generate_locations(num_users, self.area_size)
        self.user_bs_connections = self._precalculate_connections()

        self.NO_OP_ACTION_INDEX = 0
        self.action_spaces_per_bs = [size * self.num_contents + 1 for size in self.cache_sizes]
        self.n_features_per_agent_obs = 3 * self.num_contents
        self.global_obs_dim = self.num_base_stations * self.n_features_per_agent_obs

        self.cur_epoch = 0
        self.bs_caches: List[List[int]] = []
        self.bs_used_times: List[List[int]] = []
        self.bs_access_counts: List[List[int]] = []
        self.bs_cached_times: List[List[int]] = []
        # 频率窗口：deque[frozenset[int]]，每个元素代表一个时间步的“去重请求集合”
        self.bs_resource_freqs: List[Dict[str, deque]] = []
        self.ewma_reward_value = 0.0

        # 新增：记录最近一次“真正写入频率”的时间步（用于防止重复/漏记）
        self._last_freq_epoch: int = -1

        # 注意：__init__ 仅调用 reset，预填充由外部脚本在 reset 后显式调用
        self.reset(reset_caches=True, reset_freqs=True)
        _logger.info("UnifiedMultiBSCacheEnv 初始化成功。")

    def _generate_bs_locations(self, count: int, area_size: Tuple[float, float]) -> List[Tuple[float, float]]:
        center_x, center_y = area_size[0] / 2.0, area_size[1] / 2.0
        locations = [
            (center_x - 2, center_y),
            (center_x + 2, center_y),
            (center_x, center_y),
            (center_x - 2, center_y + 2),
            (center_x + 2, center_y + 2),
        ]
        return locations[:count]

    def _generate_locations(self, count: int, area_size: Tuple[float, float]) -> List[Tuple[float, float]]:
        return [(random.uniform(0, area_size[0]), random.uniform(0, area_size[1])) for _ in range(count)]

    def _precalculate_connections(self) -> List[List[int]]:
        user_bs_connections = [[] for _ in range(self.num_users)]
        for u_idx, user_loc in enumerate(self.user_locations):
            for bs_idx, bs_loc in enumerate(self.bs_locations):
                if euclidean_distance(user_loc, bs_loc) <= self.cell_radius_km:
                    user_bs_connections[u_idx].append(bs_idx)
            if not user_bs_connections[u_idx]:
                nearest_bs = min(range(self.num_base_stations), key=lambda i: euclidean_distance(user_loc, self.bs_locations[i]))
                user_bs_connections[u_idx].append(nearest_bs)
        return user_bs_connections

    def reset(self, reset_caches: bool = True, reset_freqs: bool = True):
        """
        reset 回归其本职：只将环境恢复到初始的、空的状态。
        """
        self.cur_epoch = 0
        if reset_caches:
            self.bs_caches = [[-1] * size for size in self.cache_sizes]
            self.bs_used_times = [[0] * size for size in self.cache_sizes]     # 记“最近一次被请求”的时间
            self.bs_access_counts = [[0] * size for size in self.cache_sizes]
            self.bs_cached_times = [[0] * size for size in self.cache_sizes]
        if reset_freqs:
            self.bs_resource_freqs = [
                {"short": deque(maxlen=FEAT_TERMS[0]), "mid": deque(maxlen=FEAT_TERMS[1]), "long": deque(maxlen=FEAT_TERMS[2])}
                for _ in range(self.num_base_stations)
            ]
        self._last_freq_epoch = -1
        self.ewma_reward_value = 0.0
        return self._get_obs_and_mask()

    def prefill_cache_with_expert_policy(
        self,
        expert_fn: Callable[['UnifiedMultiBSCacheEnv', int, Set[int], Any], int],
        cache_validator: CacheActionValidator,
        warmup_steps: int = 100
    ):
        """
        使用专家策略进行缓存预填充：
        1) 预热期间频率写入 temp（每步一个集合），结束后顺序合并到正式频率；
        2) 把 cur_epoch 对齐到 warmup 末尾（默认100；若数据不足则至 num_epochs-2）；
        3) 用预热期内的“请求记录”恢复每个缓存内容的最近请求时间（bs_used_times），使“未请求时长”与频率一致；
        4) 将 _last_freq_epoch 对齐为 warmup-1，表示当前 t 还未写入。
        """
        _logger.info(f"正在使用专家策略进行缓存预填充 ({warmup_steps} 步)...")

        # 预热步数（不越界）
        warmup_loop = max(0, min(warmup_steps, self.num_epochs - 1))

        # 备份正式频率（epoch 无需备份）
        original_freqs = self.bs_resource_freqs

        # 用临时频率承接预热产生的请求序列（避免污染正式对象）
        temp_freqs = [
            {"short": deque(maxlen=FEAT_TERMS[0]), "mid": deque(maxlen=FEAT_TERMS[1]), "long": deque(maxlen=FEAT_TERMS[2])}
            for _ in range(self.num_base_stations)
        ]
        # 记录预热期间每个 BS 每个内容最后一次出现的时间（用于还原未请求时长）
        last_seen: List[Dict[int, int]] = [dict() for _ in range(self.num_base_stations)]
        seen_counts: List[Dict[int, int]] = [defaultdict(int) for _ in range(self.num_base_stations)]

        # 让 _update_frequency_features 写到 temp
        self.bs_resource_freqs = temp_freqs

        # 预热（不消耗主频率对象；但会根据专家策略填充缓存）
        for t in range(warmup_loop):
            self.cur_epoch = t
            current_requests = self.requests[t]
            # 把 t 步的请求写入 temp 频率（短/中/长各 append 一个去重集合）
            self._update_frequency_features(current_requests)
            requested_sets = self._get_requested_contents_per_bs(current_requests)

            # 记录 last_seen / seen_counts
            for bs_idx in range(self.num_base_stations):
                for cid in requested_sets[bs_idx]:
                    last_seen[bs_idx][int(cid)] = t
                    seen_counts[bs_idx][int(cid)] += 1

            joint_actions = [expert_fn(self, bs_idx, requested_sets[bs_idx], cache_validator) for bs_idx in range(self.num_base_stations)]

            # 在真实缓存上应用这些决策（只改 cache，不推进主 epoch）
            for bs_idx, action in enumerate(joint_actions):
                if action == self.NO_OP_ACTION_INDEX:
                    continue
                action_idx_shifted = action - 1
                slot = action_idx_shifted // self.num_contents
                content = action_idx_shifted % self.num_contents
                if 0 <= slot < self.cache_sizes[bs_idx]:
                    self.bs_caches[bs_idx][slot] = int(content)
                    self.bs_cached_times[bs_idx][slot] = t  # 可选：记录替换发生时间

        # 把 temp 频率并入正式频率（保持时间顺序）
        self.bs_resource_freqs = original_freqs
        for bs_idx in range(self.num_base_stations):
            for term in ("short", "mid", "long"):
                for step_set in temp_freqs[bs_idx][term]:  # step_set: frozenset[int]
                    self.bs_resource_freqs[bs_idx][term].append(step_set)

        # === 对齐时间步到 warmup 末尾 ===
        warmup_epoch = min(warmup_loop, max(0, self.num_epochs - 2))
        self.cur_epoch = warmup_epoch

        # === 用预热期请求还原 bs_used_times，使“未请求时长”真实可用 ===
        for bs_idx in range(self.num_base_stations):
            for i, cid in enumerate(self.bs_caches[bs_idx]):
                if cid != -1:
                    last_t = last_seen[bs_idx].get(int(cid), 0)
                    self.bs_used_times[bs_idx][i] = last_t
                    self.bs_access_counts[bs_idx][i] = max(self.bs_access_counts[bs_idx][i], seen_counts[bs_idx].get(int(cid), 0))

        # 预热并入了 [0..warmup-1]，当前 t=warmup 尚未写
        self._last_freq_epoch = self.cur_epoch - 1

        _logger.info(f"专家策略预填充完成（频率已并入，当前时间步={self.cur_epoch}）。")

    def _update_frequency_features(self, epoch_requests: List[int]):
        """
        将本时间步的“按 BS 去重的请求集合”写入三个窗口（短/中/长）。
        每个窗口 append 一次（一个时间步 = 一个集合），避免用户数量导致的挤出效应。
        """
        requested_sets = self._get_requested_contents_per_bs(epoch_requests)
        for bs_idx in range(self.num_base_stations):
            step_set = frozenset(requested_sets[bs_idx])  # 每步只存 1 个集合
            self.bs_resource_freqs[bs_idx]["short"].append(step_set)
            self.bs_resource_freqs[bs_idx]["mid"].append(step_set)
            self.bs_resource_freqs[bs_idx]["long"].append(step_set)

    def _process_requests_and_get_reward(self, epoch_requests: List[int]) -> float:
        hit_count, total_requests = 0, 0
        for user_id, req_content in enumerate(epoch_requests):
            if req_content == -1:
                continue
            total_requests += 1
            user_request_hit = False
            if user_id < len(self.user_bs_connections):
                for bs_idx in self.user_bs_connections[user_id]:
                    if req_content in self.bs_caches[bs_idx]:
                        user_request_hit = True
                        try:
                            slot_id = self.bs_caches[bs_idx].index(req_content)
                            if 0 <= slot_id < len(self.bs_used_times[bs_idx]):
                                # 命中 -> 更新最近请求时间（用于“未请求时长”）
                                self.bs_used_times[bs_idx][slot_id] = self.cur_epoch
                                self.bs_access_counts[bs_idx][slot_id] += 1
                        except ValueError:
                            _logger.error(f"逻辑错误: 内容 {req_content} 在缓存中但 .index() 找不到。")
                        break  # 命中一次即可
            if user_request_hit:
                hit_count += 1
        return (hit_count / total_requests) if total_requests > 0 else 0.0

    def _get_bs_features(self, bs_idx: int) -> np.ndarray:
        """
        基站特征：对每个窗口，统计“该内容出现在多少个时间步”的计数。
        归一化（可选）在三个窗口的拼接向量上就地完成。
        """
        features = np.zeros(self.n_features_per_agent_obs, dtype=np.float32)
        for i, term in enumerate(["short", "mid", "long"]):
            deq = self.bs_resource_freqs[bs_idx][term]   # deque[frozenset[int]]
            counts = np.zeros(self.num_contents, dtype=np.float32)
            for step_set in deq:                         # 按时间步累计
                for cid in step_set:
                    if 0 <= cid < self.num_contents:
                        counts[cid] += 1.0
            features[i * self.num_contents : (i + 1) * self.num_contents] = counts
        max_feat_val = np.max(features)
        if self.normalize and max_feat_val > 1e-6:
            features /= max_feat_val
        return features

    def _get_observations(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        individual_obs = [self._get_bs_features(i) for i in range(self.num_base_stations)]
        global_obs = np.concatenate(individual_obs)
        return global_obs, individual_obs

    def _get_requested_contents_per_bs(self, epoch_requests: List[int]) -> List[Set[int]]:
        bs_requested_contents = [set() for _ in range(self.num_base_stations)]
        for user_id, req_content in enumerate(epoch_requests):
            if req_content != -1 and user_id < len(self.user_bs_connections):
                for bs_idx in self.user_bs_connections[user_id]:
                    bs_requested_contents[bs_idx].add(req_content)
        return bs_requested_contents

    def _get_action_masks(self, requested_contents_per_bs: List[Set[int]]) -> List[np.ndarray]:
        action_masks = []
        for bs_idx in range(self.num_base_stations):
            cache, cache_size = self.bs_caches[bs_idx], self.cache_sizes[bs_idx]
            action_space_size = self.action_spaces_per_bs[bs_idx]
            mask = np.zeros(action_space_size, dtype=bool)
            mask[self.NO_OP_ACTION_INDEX] = True
            contents_to_cache = [c for c in requested_contents_per_bs[bs_idx] if c not in cache]
            for slot_to_evict in range(cache_size):
                for content_id in contents_to_cache:
                    action_idx = slot_to_evict * self.num_contents + content_id + 1
                    if 0 < action_idx < action_space_size:
                        mask[action_idx] = True
            action_masks.append(mask)
        return action_masks

    def _get_obs_and_mask(self):
        if self.cur_epoch >= self.num_epochs:
            _logger.warning("尝试在数据结束后获取状态，返回空状态。")
            global_obs = np.zeros(self.global_obs_dim, dtype=np.float32)
            ind_obs = [np.zeros(self.n_features_per_agent_obs, dtype=np.float32) for _ in range(self.num_base_stations)]
            masks = [np.zeros(space, dtype=bool) for space in self.action_spaces_per_bs]
            return global_obs, ind_obs, masks
        global_obs, individual_obs = self._get_observations()
        current_requests = self.requests[self.cur_epoch]
        current_requested_contents = self._get_requested_contents_per_bs(current_requests)
        action_masks = self._get_action_masks(current_requested_contents)
        return global_obs, individual_obs, action_masks

    def step(self, joint_actions: List[int]):
        # 若已到末尾，返回空状态
        if self.cur_epoch >= self.num_epochs - 1:
            empty_global_obs = np.zeros(self.global_obs_dim, dtype=np.float32)
            empty_ind_obs = [np.zeros(self.n_features_per_agent_obs, dtype=np.float32) for _ in range(self.num_base_stations)]
            return empty_global_obs, empty_ind_obs, [0.0] * self.num_base_stations, True, {}

        # 在 t 时刻应用动作（只改 cache）
        for bs_idx, action in enumerate(joint_actions):
            if action == self.NO_OP_ACTION_INDEX:
                continue
            action_idx_shifted = action - 1
            slot, content = action_idx_shifted // self.num_contents, action_idx_shifted % self.num_contents
            if not (0 <= slot < self.cache_sizes[bs_idx] and 0 <= content < self.num_contents):
                _logger.warning(f"Epoch {self.cur_epoch}: BS {bs_idx} chose invalid action {action}. Ignored.")
                continue
            self.bs_caches[bs_idx][slot] = int(content)
            # 替换发生的时间（“最近一次被请求”由真实命中更新；这里不强行置为当前时间，避免虚假 age=0）
            self.bs_cached_times[bs_idx][slot] = self.cur_epoch
            self.bs_access_counts[bs_idx][slot] = max(1, self.bs_access_counts[bs_idx][slot])

        # 推进到 t+1
        self.cur_epoch += 1

        # 计算 t+1 的即时命中率（命中会更新 bs_used_times，用于 age）
        requests_t_plus_1 = self.requests[self.cur_epoch]
        instant_hit_rate = self._process_requests_and_get_reward(requests_t_plus_1)
        reward = instant_hit_rate

        # EWMA
        if self.cur_epoch == 1:
            self.ewma_reward_value = reward
        else:
            self.ewma_reward_value = self.reward_smoothing_alpha * reward + (1 - self.reward_smoothing_alpha) * self.ewma_reward_value

        # 把 t+1 写入频率（每步集合）并记录已写时间戳
        self._update_frequency_features(requests_t_plus_1)
        self._last_freq_epoch = self.cur_epoch

        next_global_obs, next_individual_obs = self._get_observations()
        done = self.cur_epoch >= self.num_epochs - 1
        info = {"instant_hit_rate": float(instant_hit_rate)}
        rewards_per_agent = [float(self.ewma_reward_value)] * self.num_base_stations
        return next_global_obs, next_individual_obs, rewards_per_agent, done, info

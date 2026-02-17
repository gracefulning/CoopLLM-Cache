# 文件名: unified_data_loader.py

import numpy as np
import pandas as pd
import logging
from typing import List, Optional

# 设置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)

class DataLoader:
    """
    基础数据加载器类。
    所有具体的数据加载器都应继承此类。
    """
    def __init__(self):
        self.requests: List[List[int]] = []  # 存储请求序列, 格式: [[u1_req, u2_req, ...], [u1_req, u2_req, ...], ...]
        self.operations: List[List[int]] = [] # 存储操作序列 (如读/写)

    def get_requests(self) -> List[List[int]]:
        """
        获取所有时间步的请求序列。
        Returns:
            一个嵌套列表，外层列表代表时间步，内层列表代表该时间步所有用户的请求。
        """
        return self.requests

    def get_operations(self) -> List[List[int]]:
        """
        获取所有时间步的操作序列。
        Returns:
            一个嵌套列表，外层列表代表时间步，内层列表代表该时间步所有用户的操作。
        """
        return self.operations

    def get_num_epochs(self) -> int:
        """获取总的时间步（回合）数。"""
        return len(self.requests)

class MultiUserDataLoaderPintos(DataLoader):
    """
    从 Pintos 跟踪文件加载多用户请求数据的数据加载器。
    (此部分代码作为可选功能保留)
    """
    def __init__(self, prog_files: List[str], num_epochs: Optional[int] = None, boot: bool = False):
        """
        初始化 Pintos 数据加载器。
        Args:
            prog_files: Pintos 跟踪文件路径列表，每个文件代表一个用户的跟踪。
            num_epochs: 指定加载的回合数。如果为 None，则取所有用户跟踪的最小长度。
            boot: 是否包含启动/执行阶段的数据。
        """
        super().__init__()
        if not prog_files:
            raise ValueError("prog_files list cannot be empty.")
            
        _logger.info(f"Loading Pintos data from {len(prog_files)} files...")
        self.num_users = len(prog_files)
        user_traces, user_ops = [], []
        min_len = float('inf')

        for prog in prog_files:
            df = pd.read_csv(prog, header=0, encoding='utf-8')
            if not boot:
                df = df.loc[df['boot/exec'] == 1, :]
            trace = list(df['blocksector'])
            ops = list(df['read/write'])
            user_traces.append(trace)
            user_ops.append(ops)
            min_len = min(min_len, len(trace))
        
        self.num_epochs = min(min_len, num_epochs) if num_epochs is not None else min_len

        self.requests = [[] for _ in range(self.num_epochs)]
        self.operations = [[] for _ in range(self.num_epochs)]
        for epoch in range(self.num_epochs):
            for user_idx in range(self.num_users):
                # 确保即使某个用户的跟踪较短，也能安全填充
                req = int(user_traces[user_idx][epoch]) if epoch < len(user_traces[user_idx]) else -1
                op = int(user_ops[user_idx][epoch]) if epoch < len(user_ops[user_idx]) else -1
                self.requests[epoch].append(req)
                self.operations[epoch].append(op)
        _logger.info(f"Pintos data loaded for {self.num_epochs} epochs.")


class MultiUserDataLoaderZipf(DataLoader):
    """
    使用 Zipf 分布生成多用户请求数据的数据加载器，支持用户分组。
    此实现确保组内用户具有相似的内容偏好，而组间偏好不同。
    """
    def __init__(self, num_files: int, num_epochs: int, num_users: int, param: float, 
                 num_groups: int = 3, operation: str = 'random', seed: Optional[int] = None):
        """
        初始化 Zipf 数据加载器。
        Args:
            num_files: 总文件数量 M。
            num_epochs: 总时间步 (T)。
            num_users: 用户数量 U。
            param: Zipf 分布参数 beta。
            num_groups: 用户分组数。
            operation: 操作类型，'random' 表示随机读写，'0' 表示只读，'1' 表示只写。
            seed: 随机种子，用于确保可复现性。
        """
        super().__init__()
        _logger.info(f"Generating Zipf data: files={num_files}, epochs={num_epochs}, users={num_users}, "
                     f"zipf_param={param}, groups={num_groups}, seed={seed}...")
        
        if seed is not None:
            np.random.seed(seed)

        self.num_epochs = num_epochs
        self.num_users = num_users
        self.num_groups = num_groups
        self.num_files = num_files
        self.param = param

        # 1. 将用户分配到不同的组
        user_groups = [[] for _ in range(num_groups)]
        for user_idx in range(num_users):
            user_groups[user_idx % num_groups].append(user_idx)

        # 2. 为每个用户组生成一个独特的 Zipf 概率分布
        group_zipf_probabilities = []
        for g_idx in range(num_groups):
            # 为每个组生成一个随机打乱的文件排名，这是实现“组间偏好不同”的关键
            shuffled_content_ids = np.arange(num_files)
            np.random.shuffle(shuffled_content_ids)

            # 计算标准的 Zipf 概率
            ranks = np.arange(1, num_files + 1)
            raw_probs = 1.0 / (ranks ** self.param)
            normalized_probs = raw_probs / np.sum(raw_probs)

            # 将概率根据打乱后的排名映射回原始内容ID
            group_prob_dist = np.zeros(num_files)
            for rank_idx, content_id in enumerate(shuffled_content_ids):
                group_prob_dist[content_id] = normalized_probs[rank_idx]
            
            group_zipf_probabilities.append(group_prob_dist)

        # 3. 为每个用户分配其所属组的 Zipf 概率分布
        user_zipf_probabilities = [None] * num_users
        for g_idx, group_users in enumerate(user_groups):
            for user_idx in group_users:
                user_zipf_probabilities[user_idx] = group_zipf_probabilities[g_idx]

        # 4. 为每个用户生成请求和操作序列，然后按时间步重组成最终格式
        self.requests = [[] for _ in range(num_epochs)]
        self.operations = [[] for _ in range(num_epochs)]

        for user_idx in range(num_users):
            # 为单个用户一次性生成所有时间步的请求
            user_reqs = np.random.choice(
                a=np.arange(num_files),
                size=num_epochs,
                p=user_zipf_probabilities[user_idx]
            )
            
            # 生成操作序列
            if operation == 'random':
                user_ops = np.random.choice([0, 1], size=num_epochs)
            else:
                user_ops = np.full(num_epochs, int(operation))

            # 将该用户的序列分配到每个时间步中
            for epoch in range(num_epochs):
                self.requests[epoch].append(user_reqs[epoch])
                self.operations[epoch].append(user_ops[epoch])
            
        _logger.info("Zipf data generation complete.")
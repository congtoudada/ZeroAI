import os
import faiss
import numpy
import numpy as np
from loguru import logger


class FaissReidHelper:
    def __init__(self, dimension, refresh_mode=0, refresh_interval=54000, refresh_count=10000):
        """
        faiss注意点：
        1.索引库下标从0开始
        2.在 faiss 中，当你删除一个向量时（例如通过 remove_ids() 方法），删除操作并不会导致索引中的后续 ID 发生变更。
          也就是说，删除 idx=1 后，其他索引（例如 idx=2 及以后的索引）并不会被修改或重新编号，它们仍然保持原来的 ID。
        :param dimension: 特征维度
        :param refresh_mode: 0:基于时间刷新 1:基于特征数量刷新 2:综合刷新(满足其中一项就刷新)
        :param refresh_interval: 经过n帧刷新一次特征库（30fps * 60s * 30min = 54000 frame）
        :param refresh_count: 达到n条数据刷新一次特征库
        """
        self.pname = f"[ {os.getpid()}:faiss_reid ]"
        self.dimension = dimension
        self.refresh_mode = refresh_mode
        self.refresh_interval = refresh_interval
        self.refresh_count = refresh_count
        self.upper_indices = []  # 上半区特征库索引
        self.down_indices = []  # 下半区特征库索引
        self.is_upper = True  # 是否激活上半区
        self.activate_database = faiss.index_factory(self.dimension, "Flat", faiss.METRIC_INNER_PRODUCT)  # 特征库
        self.activate_dict = {}  # 特征库索引字典，存储额外信息
        self.last_refresh = 0  # 上次刷新帧

    def add(self, feat, info_dict) -> int:
        """
        :param feat: 特征向量
        :param info_dict: 特征携带的信息（图片路径、摄像头id等）
        """
        # 断言 feat 的形状是 (1, d)
        assert feat.shape == (1, self.dimension), \
            f"Expected feat to have shape (1, {self.dimension}), but got {feat.shape}"
        faiss.normalize_L2(feat)
        self.activate_database.add(feat)
        idx = self.activate_database.ntotal - 1
        self.activate_dict[idx] = info_dict
        info_dict['index'] = idx
        if self.is_upper:
            self.upper_indices.append(idx)
        else:
            self.down_indices.append(idx)
        return idx

    def remove(self, idx):
        if self.activate_dict.__contains__(idx):
            ids_to_remove = np.array([idx])
            self.activate_database.remove_ids(ids_to_remove)
            self.activate_dict.pop(idx)

    def remove_range(self, ids):
        if len(ids) == 0:
            return
        ids_to_remove = np.array(ids)
        self.activate_database.remove_ids(ids_to_remove)
        for i, idx in enumerate(ids):
            if self.activate_dict.__contains__(idx):
                self.activate_dict.pop(idx)

    def get_total(self):
        return self.activate_database.ntotal

    def search(self, query, top_k=4):
        assert query.shape == (1, self.dimension), \
            f"Expected feat to have shape (1, {self.dimension}), but got {query.shape}"
        D, I = self.activate_database.search(query, k=top_k)
        logger.info(f"{self.pname} 查询结果: \nI: {I} \nD: {D}")
        if I[0][0] == -1:
            return []
        # values = [self.activate_dict[key] for key in I.flatten().tolist()]
        values_with_scores = [
            {**self.activate_dict[key], 'score': D[0][i]}  # 合并字典并添加 score 键
            for i, key in enumerate(I.flatten().tolist())
        ]
        return values_with_scores

    def tick(self, now):
        if self.last_refresh == 0:  # 第一次tick不刷新
            self.last_refresh = now
            return
        if self.refresh_mode == 0:
            if abs(now - self.last_refresh) > self.refresh_interval:
                self.refresh(now)
        elif self.refresh_mode == 1:
            if self.activate_database.ntotal > self.refresh_count:
                self.refresh(now)
        else:
            if self.activate_database.ntotal > self.refresh_count or now - self.last_refresh > self.refresh_interval:
                self.refresh(now)

    def refresh(self, now):
        logger.info(f"{self.pname} before switch total: {self.get_total()}")
        # 刷新半区
        if self.is_upper:
            # 清空下半区数据
            self.remove_range(self.down_indices)
            self.down_indices.clear()
            # 切换成下半区
            self.is_upper = False
            logger.info(f"{self.pname} after switch to down, current total: {self.get_total()}")
        else:
            # 清空上半区数据
            self.remove_range(self.upper_indices)
            self.upper_indices.clear()
            # 切换成上半区
            self.is_upper = True
            logger.info(f"{self.pname} switch to upper, current total: {self.get_total()}")
        self.last_refresh = now

    def destroy(self):
        self.activate_database.reset()
        self.activate_dict.clear()
        self.upper_indices.clear()
        self.down_indices.clear()

import os

import faiss
import numpy
import numpy as np
from loguru import logger

from clip_reid.zero.helper.faiss_reid_info import FaissReidInfo
from utility.config_kit import ConfigKit


class FaissReidHelper:
    def __init__(self, dimension, refresh_interval=54000):
        """
        faiss注意点：
        1.索引库下标从0开始
        2.在 faiss 中，当你删除一个向量时（例如通过 remove_ids() 方法），删除操作并不会导致索引中的后续 ID 发生变更。
          也就是说，删除 idx=1 后，其他索引（例如 idx=2 及以后的索引）并不会被修改或重新编号，它们仍然保持原来的 ID。
        3.refresh_interval = 30fps * 60s * 30min = 54000 frame
        """
        self.pname = f"[ {os.getpid()}:faiss_reid ]"
        self.dimension = dimension
        self.refresh_interval = refresh_interval
        self.upper_database = faiss.index_factory(self.dimension, "Flat", faiss.METRIC_INNER_PRODUCT)  # 上半区特征库
        self.upper_dict = {}
        self.down_database = faiss.index_factory(self.dimension, "Flat", faiss.METRIC_INNER_PRODUCT)  # 下半区特征库
        self.down_dict = {}
        self.is_upper = True  # 是否激活上半区
        self.activate_database = self.upper_database  # 当前激活特征库
        self.activate_dict = self.upper_dict  # 当前激活字典
        self.map = {}
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
        return idx

    def search(self, query, top_k=4):
        assert query.shape == (1, self.dimension), \
            f"Expected feat to have shape (1, {self.dimension}), but got {query.shape}"
        D, I = self.activate_database.search(query, k=top_k)
        logger.info(f"{self.pname} 查询结果: \nI: {I} \nD: {D}")
        values = [self.activate_dict[key] for key in I.flatten().tolist()]
        return values

    def tick(self, now):
        if now - self.last_refresh > self.refresh_interval:
            # 刷新半区
            if self.is_upper:
                # 清空下半区数据
                self.down_database.reset()
                self.down_dict.clear()
                # 切换半区
                self.activate_database = self.down_database
                self.activate_dict = self.down_dict
                self.is_upper = False
            else:
                # 清空上半区数据
                self.upper_database.reset()
                self.upper_dict.clear()
                # 切换半区
                self.activate_database = self.upper_database
                self.activate_dict = self.upper_dict
                self.is_upper = True
            self.last_refresh = now

    def destroy(self):
        self.activate_database = None
        self.activate_dict = None
        self.upper_dict.clear()
        self.upper_database.reset()
        self.down_database.reset()
        self.down_dict.clear()


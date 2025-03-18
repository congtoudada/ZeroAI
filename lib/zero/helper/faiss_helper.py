import os
import sys

import faiss
import numpy
import numpy as np
from loguru import logger
from datetime import datetime

from utility.timer_kit import TimerKit
from zero.helper.analysis_helper import AnalysisHelper


class FaissHelper:

    def __init__(self, dimension, refresh_mode=0, refresh_interval=54000,
                 refresh_count=10000, remove_callback=None, enable_log=True,
                 enable_analysis=False):
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
        self.remove_callback = remove_callback  # 删除回调(参数为dict,删除特征对应的extra info)
        self.upper_indices = []  # 上半区特征库索引
        self.down_indices = []  # 下半区特征库索引
        self.is_upper = True  # 是否激活上半区
        self.activate_database = faiss.index_factory(self.dimension, "Flat", faiss.METRIC_INNER_PRODUCT)  # 特征库
        self.activate_database = faiss.IndexIDMap(self.activate_database)
        self.activate_dict = {}  # 特征库索引字典，存储额外信息 key:
        self.last_refresh = 0  # 上次刷新帧
        self.index = 0  # 索引
        self.enable_log = enable_log
        self.enable_analysis = enable_analysis
        self.search_timer = TimerKit(max_flag=0)  # 匹配计时器
        self.gc_timer = TimerKit(max_flag=0)  # 回收计时器

    def add(self, feat, info_dict) -> int:
        """
        :param feat: 特征向量
        :param info_dict: 特征携带的信息（图片路径、摄像头id等）
        """
        # 断言 feat 的形状是 (1, d)
        assert feat.shape == (1, self.dimension), \
            f"Expected feat to have shape (1, {self.dimension}), but got {feat.shape}"
        faiss.normalize_L2(feat)
        # idx = self.activate_database.add(feat)
        self.index = (self.index + 1) % sys.maxsize
        idx = self.index
        self.activate_database.add_with_ids(feat, np.array([idx], dtype='int64'))
        info_dict['index'] = idx
        self.activate_dict[idx] = info_dict
        if self.is_upper:
            self.upper_indices.append(idx)
        else:
            self.down_indices.append(idx)
        return idx

    def remove(self, idx):
        ids_to_remove = np.array([idx])
        if self.activate_dict.__contains__(idx):
            self.activate_database.remove_ids(ids_to_remove)
            if self.remove_callback is not None:
                self.remove_callback(self.activate_dict[idx])
            self.activate_dict.pop(idx)

    def remove_range(self, ids):
        if len(ids) == 0:
            return
        ids_to_remove = np.array(ids)
        self.activate_database.remove_ids(ids_to_remove)
        for i, idx in enumerate(ids):
            if self.activate_dict.__contains__(idx):
                if self.remove_callback is not None:
                    self.remove_callback(self.activate_dict[idx])
                self.activate_dict.pop(idx)

    def get_total(self):
        return self.activate_database.ntotal

    def search(self, query, top_k=4, conf=0, sort=False):
        if top_k == 0:
            return []
        assert query.shape == (1, self.dimension), \
            f"Expected feat to have shape (1, {self.dimension}), but got {query.shape}"
        faiss.normalize_L2(query)
        self.search_timer.tic()
        D, I = self.activate_database.search(query, k=top_k)
        self.search_timer.toc()
        if I[0][0] == -1:
            return []
        # values = [self.activate_dict[key] for key in I.flatten().tolist()]
        values_with_scores = [
            {**self.activate_dict[key], 'score': float(D[0][i])}  # 合并字典并添加 score 键
            for i, key in enumerate(I.flatten().tolist())
            if float(D[0][i]) > conf
        ]
        # 额外补充时间和cam_id key-value
        for item in values_with_scores:
            if item.__contains__('img_path'):
                img_path = item["img_path"]
                cam_id = img_path.split('_')[-1].split('.')[0]
                time_str = img_path.split('_')[-2]
                item['cam_id'] = cam_id
                item['recordTime'] = time_str
                item['shot_img'] = os.path.abspath(img_path)
            else:
                logger.error(f"{self.pname} 搜索结果集找不到key: img_path")
        if self.enable_log:
            logger.info(f"{self.pname} 查询结果: \nI: {I} \nD: {D} \nExtra: {values_with_scores}")
        if self.enable_analysis:
            AnalysisHelper.refresh(f"{self.pname} Search max time", self.search_timer.max_time * 1000, 33.3)
            AnalysisHelper.refresh(f"{self.pname} Search average time", self.search_timer.average_time * 1000, 33.3)

        if sort:
            values_with_scores = sorted(
                values_with_scores,
                key=lambda x: datetime.strptime(x['recordTime'], "%Y-%m-%d-%H-%M-%S")
            )
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
        if self.enable_log:
            logger.info(f"{self.pname} before switch total: {self.get_total()}")
        self.gc_timer.tic()
        # 刷新半区
        if self.is_upper:
            # 清空下半区数据
            self.remove_range(self.down_indices)
            self.down_indices.clear()
            # 切换成下半区
            self.is_upper = False
            if self.enable_log:
                logger.info(f"{self.pname} after switch to down, current total: {self.get_total()}")
        else:
            # 清空上半区数据
            self.remove_range(self.upper_indices)
            self.upper_indices.clear()
            # 切换成上半区
            self.is_upper = True
            if self.enable_log:
                logger.info(f"{self.pname} switch to upper, current total: {self.get_total()}")
        self.last_refresh = now
        self.gc_timer.toc()
        if self.enable_analysis:
            AnalysisHelper.refresh(f"{self.pname} Refresh Database average time", self.gc_timer.average_time * 1000, 100)

    def destroy(self):
        self.activate_database.reset()
        self.activate_dict.clear()
        self.upper_indices.clear()
        self.down_indices.clear()


from abc import ABC, abstractmethod


class IVadObjWrapper(ABC):

    @abstractmethod
    def inference_batch(self, frames_batch, objs_info_batch):
        """
        输入图像batch，返回分数
        :param frames_batch: 输入图像 List[ndarray(n,h,w,3)] BGR n由算法决定 len=batch
        :param objs_info_batch: 对象信息 List[ndarray(n,7)]
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id  （可选）
        """
        pass

    @abstractmethod
    def inference(self, frames, objs_info):
        """
        输入图像，返回分数
        :param frames: 输入图像 ndarray(n,h,w,3) BGR n由算法决定
        :param objs_info: 对象信息 List[ndarray(n,7)]
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id  （可选）
        """
        pass


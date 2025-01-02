from abc import ABC, abstractmethod


class IDetectionWrapper(ABC):
    @abstractmethod
    def inference(self, img):
        """
        输入图像，返回检测结果，格式要求
        # ndarray shape: [n, 6]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于img分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)

        :param img: 输入图像(ndarray格式)
        """
        pass


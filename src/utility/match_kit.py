from typing import List


class DetectionRecord:
    """
    用于临时缓存Object Detection结果
    """

    def __init__(self):
        self.obj_id = -1  # 对象id（可选）
        self.ltrb = None  # 包围盒
        self.score = 0  # 置信度
        self.cls = -1  # 类别
        self.has_match = False  # 是否已匹配

    def init(self, ltrb, score, cls, obj_id=-1):
        self.ltrb = ltrb
        self.score = score
        self.cls = cls
        self.has_match = False  # 是否已匹配
        self.obj_id = obj_id


class MatchKit(object):
    @staticmethod
    def match_bboxes(main_boxes: List[DetectionRecord], sub_boxes: List[DetectionRecord],
                     tolerance: int = 5) -> (List[int], List[int]):
        """
        基于包围盒匹配: main_box需在sub_box内，容差为tolerance
        匹配成功返回各自obj_id（-1代表匹配失败）
        TODO: 如果目标对象很多，可在比较过程中删除记录，加速这个过程
        """
        main_list = []
        sub_list = []
        for i, main_box in enumerate(main_boxes):
            if main_box.has_match:
                continue
            for j, sub_box in enumerate(sub_boxes):
                if sub_box.has_match:
                    continue
                if MatchKit._match_bbox(main_box, sub_box, tolerance):
                    main_box.has_match = True
                    sub_box.has_match = True
                    main_list.append(i)
                    sub_list.append(j)
        return main_list, sub_list

    @staticmethod
    def _match_bbox(main_box, sub_box, tolerance) -> bool:
        """
        包围盒匹配
        """
        main_x = (main_box.ltrb[0] + main_box.ltrb[2]) / 2.
        main_y = (main_box.ltrb[1] + main_box.ltrb[3]) / 2.
        if (sub_box.ltrb[0] - tolerance < main_x < sub_box.ltrb[2] + tolerance and
                sub_box.ltrb[1] - tolerance < main_y < sub_box.ltrb[3] + tolerance):
            return True
        else:
            return False

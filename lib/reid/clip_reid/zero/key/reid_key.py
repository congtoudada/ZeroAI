from enum import Enum


class ReidKey(Enum):
    """
    人脸识别Key
    使用举例: REIDKey.REID_REQ.name
    """
    # --- 实际请求key REQ ---
    REID_REQ = 0  # Reid请求Key
    REID_REQ_CAM_ID = 1  # 请求摄像头id
    REID_REQ_PID = 2  # 请求pid
    REID_REQ_OBJ_ID = 4  # 请求对象id
    REID_REQ_IMAGE = 5  # 请求图像
    REID_REQ_METHOD = 6  # 请求方式 0:存图请求 1:Reid请求 2:找人请求
    # --- 实际响应key RSP + port + pid ---
    REID_RSP = 10
    REID_RSP_OBJ_ID = 11  # 对象id
    REID_RSP_PER_ID = 12  # Reid识别结果（1为陌生人）
    REID_RSP_SCORE = 13  # Reid识别分数(0~1)


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
    REID_REQ_METHOD = 6  # 请求方式 1:存图请求 2:Reid请求 3:找人请求
    REID_REQ_STATUS = 7  # 报警类型（可选） 1:Phone 2:Helmet 3:Card 4:Intrude
    # --- 实际响应key RSP + port + pid ---
    REID_RSP = 10  # Reid识别响应Key 对应METHOD1
    REID_RSP_OBJ_ID = 11  # 对象id
    REID_RSP_PER_ID = 12  # Reid识别结果（1为陌生人）
    REID_RSP_SCORE = 13  # Reid识别分数(0~1)
    # 找人
    REID_RSP_SP = 20  # SP<=>Search Person 找人响应key 对应METHOD2
    REID_RSP_SP_PACKAGE = 21  # 响应包，返回List[Dict{"cam_id":xxx, "path": xxx, "score": xxx}]
    # REID_RSP_CAM_IDS = 22  # 摄像头ids
    # REID_RSP_SCORES = 23  # 识别分数
    # REID_RSP_IMAGES = 24  # 图片路径


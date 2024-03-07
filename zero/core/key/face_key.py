from enum import Enum


class FaceKey(Enum):
    """
    人脸识别Key
    """
    # --- 实际请求key REQ + port ---
    REQ = 0
    REQ_CAM_ID = 1
    REQ_PID = 2  #
    REQ_OBJ_ID = 4  # 追踪对象的id
    REQ_IMAGE = 5  # 人脸图像
    # --- 实际响应key RSP + port + pid ---
    RSP = 10
    RSP_OBJ_ID = 11
    RSP_PER_ID = 12  # 人脸识别结果（1为陌生人）
    RSP_SCORE = 13  # 人脸识别分数


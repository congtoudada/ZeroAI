from enum import Enum


class FaceKey(Enum):
    """
    人脸识别Key
    使用举例: FaceKey.FACE_REQ.name
    """
    # --- 实际请求key REQ ---
    FACE_REQ = 0  # 人脸识别请求Key
    FACE_REQ_CAM_ID = 1  # 请求摄像头id
    FACE_REQ_PID = 2  # 请求pid
    FACE_REQ_OBJ_ID = 4  # 请求对象id
    FACE_REQ_IMAGE = 5  # 请求图像
    # --- 实际响应key RSP + port + pid ---
    FACE_RSP = 10
    FACE_RSP_OBJ_ID = 11
    FACE_RSP_PER_ID = 12  # 人脸识别结果（1为陌生人）
    FACE_RSP_SCORE = 13  # 人脸识别分数


import os
import time
from loguru import logger
import cv2

from zero.utility.web_kit import WebKit
from enum import Enum


# class WarnType(Enum):
#     Phone = 1,
#     Helmet = 2,
#     Card = 3,
#     Intrude = 4

class WarnKit:
    @staticmethod
    def send_warn_result(pname, output_dir, camId, warnType, per_id, shot_img, img_enable, web_enable):
        # 导出图
        time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        warn_str = ""
        if warnType == 1:
            warn_str = "Phone"
        elif warnType == 2:
            warn_str = "Helmet"
        elif warnType == 3:
            warn_str = "Card"
        elif warnType == 4:
            warn_str = "Intrude"
        img_path = os.path.join(output_dir, f"{time_str}_{warn_str}_{per_id}.jpg")
        if img_enable:
            cv2.imwrite(img_path, shot_img)
            logger.info(f"{pname}存图成功，路径: {img_path}")
        if web_enable:
            # 通知后端
            data = {
                "recordTime": time_str,
                "camId": camId,
                "warnType":  warnType,
                "personId": per_id,
                "shotImg": img_path
            }
            WebKit.post(f"{WebKit.Prefix_url}/warn", data)
            logger.info(f"{pname}发送后端请求，路径: {WebKit.Prefix_url}/warn")


# if __name__ == "__main__":
#     warnType = WarnType.Phone
#     print(warnType.value[0])

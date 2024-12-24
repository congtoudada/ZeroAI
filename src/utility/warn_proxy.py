import os
import time
import cv2
from loguru import logger

from simple_http.simple_http_helper import SimpleHttpHelper


class WarnProxy(object):
    @staticmethod
    def send(http_helper: SimpleHttpHelper, pname, output_dir, camId, warnType,
             per_id, shot_img, warn_score, img_enable, web_enable):
        """
        :param http_helper: http帮助类
        :param pname: 代理的进程名
        :param output_dir: 输出路径
        :param camId: 摄像头id
        :param warnType: 报警类型 (1:phone 2:helmet 3:card 4:intrude)
        :param per_id: 人员id id==1为陌生人
        :param shot_img: 截图
        :param warn_score: 报警得分
        :param img_enable: 是否写入本地
        :param web_enable: 是否发送web
        """
        if http_helper is None:
            logger.error(f"{pname} 发送http请求失败！http_helper为None！")
            return
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
        if img_enable and shot_img is not None:
            cv2.imwrite(img_path, shot_img)
            # logger.info(f"{pname} 存图成功，路径: {img_path}")
        if web_enable:
            # 通知后端
            data = {
                "recordTime": time_str,
                "camId": camId,
                "warnType": warnType,
                "personId": per_id,
                "shotImg": img_path,
                "warnScore": warn_score
            }
            if http_helper.config.debug_enable:
                logger.info(f"{pname} 发送数据: {data}")
            http_helper.post("/algorithm/warn", data)

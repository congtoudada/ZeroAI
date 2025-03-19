import os
import time
import traceback
from typing import Dict
import cv2
import numpy as np
from loguru import logger

from src.count.count_comp import CountComponent
from src.renlian.renlian_info import RenlianInfo
from src.renlian.renlian_item import RenlianItem
from insight.zero.component.face_helper import FaceHelper
from zero.key.global_key import GlobalKey
from utility.config_kit import ConfigKit
from utility.img_kit import ImgKit
from utility.object_pool import ObjectPool


class RenlianComponent(CountComponent):
    """
    计数的同时进行人脸识别
    """
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory, config_path)
        # 重新定义
        self.config: RenlianInfo = RenlianInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:renlian for {self.config.input_ports[0]}]"
        self.pool: ObjectPool = ObjectPool(20, RenlianItem)  # 对象池
        self.item_dict: Dict[int, RenlianItem] = {}  # 当前检测对象字典
        self.face_helper: FaceHelper = None

    def on_start(self):
        super().on_start()
        self.face_helper = FaceHelper(self.config.count_face_config, self._face_callback)

    def on_handle_stream(self, idx, frame, input_det):
        ret = super().on_handle_stream(idx, frame, input_det)
        # 人脸识别请求
        current_id = self.frame_id_cache[0]
        for key, value in self.item_dict.items():
            if len(value.red_seq) > 0:
                ret = self._get_dir(value.red_seq[0] == 0, self.config.count_reverse)
                if ret:  # 只有进入做人脸识别
                    # face 请求
                    if self.face_helper.try_send(current_id, self.frames[0], value.ltrb, key, value.base_x,
                                                 value.base_y, self.cam_id):
                        value.face_req_image = self.frames[0].copy()  # 请求成功存当前图
            # reid存图
            if not value.has_save_reid and value.per_id != 1:
                img_shot = self.crop_valid_img(frame, value)
                if img_shot is None or img_shot.shape[0] == 0 or img_shot.shape[1] == 0:
                    continue
                self.save_reid_img(img_shot, self.config.reid_path, value.per_id)
                image_path = os.path.join(self.config.reid_path, f"{value.per_id}_{self.cam_id}.jpg")
                logger.info(f"{self.pname} reid存图成功，路径: {image_path}")
                value.has_save_reid = True
            # 最优存图
            if self.config.reid_best_enable:
                if value.person_score > value.best_person_score:
                    w = value.ltrb[2] - value.ltrb[0]
                    h = value.ltrb[3] - value.ltrb[1]
                    if float(w) / h < self.config.reid_best_aspect:  # 长宽比 < aspect
                        img_shot = self.crop_valid_img(frame, value)
                        if img_shot is None or img_shot.shape[0] == 0 or img_shot.shape[1] == 0:
                            continue
                        value.best_person_image = img_shot
                        value.best_person_score = value.person_score
        return ret

    def on_update(self):
        super().on_update()
        current_id = self.frame_id_cache[0]
        # 人脸识别帮助tick，用于接受响应
        self.face_helper.tick(current_id)

    def send_result(self, frame, status: int, item: RenlianItem):
        """
        结果通知
        :param frame:
        :param status: 1进2出
        :param item:
        :return:
        """
        # 方向过滤
        if (status == 1 and self.config.count_filter == 1) or (status == 2 and self.config.count_filter == 2):
            return
        # 导出图
        time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        status_str = "In" if status == 1 else "Out"
        img_path = os.path.join(self.output_dir[0], f"{time_str}_{status_str}_{item.per_id}.jpg")
        # 如果存在人脸识别图就使用人脸识别图导出，否则使用当前帧人的截图
        if item.face_req_image is not None:
            img_shot = item.face_req_image.copy()
        else:
            img_shot = self.crop_valid_img(frame, item)
        if self.config.stream_export_img_enable:
            if img_shot is None or img_shot.shape[0] == 0 or img_shot.shape[1] == 0:
                pass
            else:
                cv2.imwrite(img_path, img_shot)
        # reid最优存图
        if self.config.reid_best_enable and self.config.reid_enable:
            if item.per_id is not None and item.best_person_image is not None:
                self.save_reid_img(item.best_person_image.copy(), self.config.reid_path, item.per_id)
                image_path = os.path.join(self.config.reid_path, f"{item.per_id}_{self.cam_id}.jpg")
                logger.info(f"{self.pname} reid最优存图成功，路径: {image_path}")
                item.has_save_reid = True
            # reid存图（报警时记录）
            # if self.config.reid_enable and item.per_id != 1:
            #     self.save_reid_img(frame, item.ltrb, self.config.reid_path, item.per_id)
            #     logger.info(f"{self.pname} reid存图成功，路径: {img_path}")

        if self.config.stream_web_enable:
            # 通知后端
            data = {
                "recordTime": time_str,
                "camId": self.cam_id,
                "status": status,
                "personId": item.per_id,
                "shotImg": os.path.abspath(img_path)
            }
            if self.http_helper.config.debug_enable:
                logger.info(f"{self.pname} 发送人脸结果: {data}")
            # WebKit.post(f"{WebKit.Prefix_url}/count", data)
            self.http_helper.post("/algorithm/face", data)

    def on_destroy_obj(self, obj_id):
        self.face_helper.destroy_obj(obj_id)

    def on_draw_vis(self, idx, frame, input_mot):
        frame = super().on_draw_vis(idx, frame, input_mot)
        # 人脸参考线
        # y
        # point1 = (0, int(self.face_helper.config.face_cull_up_y * self.stream_height))
        point2 = (self.stream_width, int(self.face_helper.config.face_cull_up_y * self.stream_height))
        # point3 = (0, int((1 - self.face_helper.config.face_cull_down_y) * self.stream_height))
        point4 = (self.stream_width, int((1 - self.face_helper.config.face_cull_down_y) * self.stream_height))
        # x
        point5 = (int(self.face_helper.config.face_cull_left_x * self.stream_width), 0)
        # point6 = (int(self.face_helper.config.face_cull_left_x * self.stream_width), self.stream_height)
        point7 = (int((1 - self.face_helper.config.face_cull_right_x) * self.stream_width), 0)
        # point8 = (int((1 - self.face_helper.config.face_cull_right_x) * self.stream_width), self.stream_height)
        # 交线
        line_up1 = (point5[0], point2[1])
        line_up2 = (point7[0], point2[1])
        line_down1 = (point5[0], point4[1])
        line_down2 = (point7[0], point4[1])
        line_left1 = (point5[0], point2[1])
        line_left2 = (point5[0], point4[1])
        line_right1 = (point7[0], point2[1])
        line_right2 = (point7[0], point4[1])
        cv2.line(frame, line_up1, line_up2, (255, 255, 0), 2)  # 绘制线条
        cv2.line(frame, line_down1, line_down2, (255, 255, 0), 2)  # 绘制线条
        cv2.line(frame, line_left1, line_left2, (255, 255, 0), 2)  # 绘制线条
        cv2.line(frame, line_right1, line_right2, (255, 255, 0), 2)  # 绘制线条
        # 人脸识别结果
        for key, value in self.item_dict.items():
            ltrb = value.ltrb
            per_id = value.per_id
            cv2.putText(frame, f"{per_id}",
                        (int((ltrb[0] + ltrb[2]) / 2), int(ltrb[1])),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        # 可视化并返回
        return frame

    def _face_callback(self, obj_id, per_id, score):
        if self.item_dict.__contains__(obj_id):
            self.item_dict[obj_id].per_id = per_id
            self.item_dict[obj_id].score = score

    def crop_valid_img(self, frame, item: RenlianItem):
        if frame is None:
            return
        img_shot = ImgKit.crop_img(frame, item.ltrb)
        return img_shot

    def save_reid_img(self, img, path='', id=1):
        # 创建目录
        if not os.path.exists(path):
            os.makedirs(path)
        # 裁剪图像
        # img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        if img.size == 0:
            logger.error(f"{self.pname} 警告: 裁剪的图像为空。")
            return None, None
        # 图片命名
        image_name = f"{id}_{self.cam_id}.jpg"
        image_path = os.path.join(path, image_name)

        # 保存图片
        try:
            logger.info(f"{self.pname} reid存图: {image_path}")
            cv2.imwrite(image_path, img)
        except Exception as e:
            print(f"错误: 保存图像失败 - {e}")
            return None, None

        return image_path, img


def create_process(shared_memory, config_path: str):
    comp: RenlianComponent = RenlianComponent(shared_memory, config_path)  # 创建组件
    try:
        comp.start()  # 初始化
        # 初始化结束通知
        shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
        while not shared_memory[GlobalKey.ALL_READY.name]:
            time.sleep(0.1)
        comp.update()  # 算法逻辑循环
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        # 使用 traceback 打印堆栈信息
        logger.error(f"RenlianComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()


if __name__ == '__main__':
    arr1 = [0, 1]
    arr2 = [0]
    arr3 = np.array(arr1)
    arr4 = np.array(arr2)
    print(arr1 == arr2)
    print(np.array_equal(arr3, arr4))

    # [0,0.5] [0.75,0.5] [1, 0.6]
    # [0.25,0.25]
    ref_vec = np.array([0.75, 0, 0, -0.25, 0.1, 0]).reshape(2, 3)
    input_vec = np.array([0.25, -0.25, 0])
    print(np.dot(ref_vec, input_vec)[np.dot(ref_vec, input_vec) < 0])
    # print(np.cross(ref_vec, input_vec))

    # ref_vec = np.array([0.75, 0, 0])
    # ref_vec = np.array([0.25, 0.1, 0])
    # print(np.cross(ref_vec, input_vec))

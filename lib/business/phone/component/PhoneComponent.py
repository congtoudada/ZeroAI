

# PhoneComponent

import os
import time
import cv2
from loguru import logger
from math import sqrt
from datetime import datetime, timedelta
import shutil
from common.warn_kit import WarnKit
from phone.info.PhoneInfo import PhoneInfo
from zero.core.component.based.based_multi_mot_comp import BasedMultiMOTComponent
from zero.utility.config_kit import ConfigKit


import glob
import requests

class PhoneComponent(BasedMultiMOTComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: PhoneInfo = PhoneInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:phone ]"
        self.counter = {}  # 记录连续识别同一对象次数
        self.det_record = {}  # key: person_id  value: last_time
        self.timing_record = None  # 记录上一次定时保存的时间
        self.state = []  # 存储已经报警的id，避免重复报警

    def on_update(self) -> bool:
        """
        # mot output shape: [n, 7]
        # n: n个对象
        # [0,1,2,3]: tlbr bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        """
        person_bboxes = []
        phone_bboxes = []
        person_id = []
        #phone_id = []
        
        phonePerson_bbox_dict = {}  # 存储拿手机人员的bbox
        allPerson_bbox_dict = {}    # 存储所有检测到的人的bbox

        # 检查父类更新状态并确认输入数据不为空
        if super().on_update() and self.input_mot is not None:
            for i in range(len(self.input_mot)):  # 遍历每个检测模型的输出
                if self.input_mot[i] is not None:
                    for j in range(len(self.input_mot[i])):  # 遍历每一个对象
                        cls = int(self.input_mot[i][j][5])
                        # 如果不是人或手机（类别0），则跳过
                            # 人和手机都是0
                        if cls != 0:
                            continue  
                        
                        obj_id = self.input_mot[i][j][6]
                        bbox = self.input_mot[i][j][:4]

                        # if self.config.input_port[i] == "camera3-bytetrack_person1": # 硬编码错误
                        if self.config.input_port[i] == "camera9-bytetrack_phone_person" or self.config.input_port[i] == "camera10-bytetrack_phone_person":
                            if cls == 0:  # 如果是人
                                person_bboxes.append(bbox)
                                person_id.append(obj_id)
                                allPerson_bbox_dict[obj_id] = bbox          # 所有检测到的人的包围框
                        
                        # elif self.config.input_port[i] == "camera3-bytetrack_phone1":  # 硬编码错误
                        elif self.config.input_port[i] == "camera9-bytetrack_phone" or self.config.input_port[i] == "camera10-bytetrack_phone":
                            if cls == 0:  # 如果是手机
                                phone_bboxes.append(bbox)
                                #phone_id.append(obj_id)
                                

                    # 检测到有人拿手机  则算并存一下拿手机人的包围框
                    if phone_bboxes:  
                        # 通过人和手机 bbox 之间的几何距离找到拿手机的人
                        output_bboxes, output_idx = self.on_calculate(person_bboxes, phone_bboxes)
                        for idx, bbox in zip(output_idx, output_bboxes):
                            phonePerson_bbox_dict[person_id[idx]] = bbox     # 拿手机的人的包围框
                            
                    
                    # 调用on_execute方法，传递所有人的边界框和拿手机的人的边界框
                    #print(allPerson_bbox_dict,"调试89")
                    self.on_execute(allPerson_bbox_dict, phonePerson_bbox_dict)

        return False

    def on_analysis(self):
        logger.info(f"{self.pname} video fps: {1. / max(1e-5, self.update_timer.average_time):.2f}"
                    f" inference fps: {1. / max(1e-5, self.timer.average_time):.2f}")


    def on_calculate(self, person_bboxes, phone_bboxes):
        """    
        返回拿手机人员的包围框框和索引列表
        通过计算人的包围框和手机包围框中心点的距离，来看这个手机是谁拿的
        """
        res_bbox = [] # 存储拿手机人员的包围框
        res_idx = [] # 存储拿手机人员的索引
        
        # 计算每个人的包围框中心点
        person_centres = []
        for person_bbox in person_bboxes:
            person_centres.append([(person_bbox[0] + person_bbox[2]) / 2, (person_bbox[1] + person_bbox[3]) / 2])
        
        # 遍历每个手机的包围框，找到最近的人
        for phone_bbox in phone_bboxes:
            phone_centre = [(phone_bbox[0] + phone_bbox[2]) / 2, (phone_bbox[1] + phone_bbox[3]) / 2]
            
            min_index = -1  # 初始化最近的人的下标
            min_dist = -1  # 初始化最小距离
            
            # for index, person_centre in enumerate(person_centres):
            #     # 如果是第一次计算或者找到更近的人，更新最小距离和索引
            #         #min_dist == -1意味着第一次计算
            #     if min_dist == -1:
            #         min_index = index
            #         min_dist = sqrt(
            #             (phone_centre[0] - person_centre[0]) ** 2 + (phone_centre[1] - person_centre[1]) ** 2)
            #     #
            #     else:
            #         dist = sqrt((phone_centre[0] - person_centre[0]) ** 2 + (phone_centre[1] - person_centre[1]) ** 2)
            #         if dist < min_dist:
            #             min_index = index
            #             min_dist = dist
            
            # # 如果找到了最近的人，则将其包围框和索引添加到结果列表中
            # if 0 <= min_index < len(person_bboxes):
            #     res_bbox.append(person_bboxes[min_index])
            #     res_idx.append(min_index)
            
            # 遍历每个人的包围框中心点，找最近的人
            for index, person_centre in enumerate(person_centres):
                # 计算当前手机中心点与当前人中心点的距离
                dist = sqrt((phone_centre[0] - person_centre[0]) ** 2 + (phone_centre[1] - person_centre[1]) ** 2)

                # 如果是第一次检查距离或者找到了更近的距离，则更新最小距离和索引
                if min_dist == -1 or dist < min_dist:
                    min_index = index
                    min_dist = dist

            # 检查是否有最近的人被找到，并更新结果列表
            if min_index != -1:
                res_bbox.append(person_bboxes[min_index])  # 添加最近人的包围框
                res_idx.append(min_index)  # 添加最近人的索引
                
        # 返回拿手机的人员的包围框和索引列表
        return res_bbox, res_idx

    def on_draw_vis(self, frame, vis=False, window_name="", is_copy=True):
        """
        # output shape: [n, 7]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        :param frame:
        :param vis:
        :param window_name:
        :param is_copy:
        :return:
        """
        if self.input_mot is not None:
            for i in range(len(self.input_mot)):  # 遍历内一个检测模型的输出
                if self.input_mot[i] is not None:
                    for obj in self.input_mot[i]:  # 遍历每一个对象
                        cls = int(obj[5])
                        if cls == 0:
                            ltrb = obj[:4]
                            obj_id = int(obj[6])
                            cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                                          color=(0, 0, 255), thickness=1)
                            # label = "person" if i == 0 else "phone"
                            label = "phone" if i == 0 else "person"
                            cv2.putText(frame, f"{obj_id}({label})",
                                        (int(ltrb[0]), int(ltrb[1])),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=1)
        # 可视化并返回
        return super().on_draw_vis(frame, vis, window_name)
    
    # 定时存图
    def on_execute(self, all_bboxes, object_bboxes):   
        
        #timming
        delta = timedelta(seconds=2)
        now = datetime.now()
        last = self.timing_record
        # 如果上次记录的时间不存在或者已经超过了设定的时间间隔，则执行存图操作
        if last is None or (now - last) >= delta:
            # 遍历所有包围框，并且如果配置允许，则对每个包围框执行保存图片的操作
            if all_bboxes:
                for id, bbox in all_bboxes.items():
                    if self.config.timing_enable:
                        self.on_save_img(bbox, id, self.config.timing_path)
            # 更新最后一次记录时间
            self.timing_record = now
        # 如果指定目录中的图片数量超过1000，则清空该目录
        if self.count_images_in_directory(self.config.timing_path) > 1000:
            self.clear_directory(self.config.timing_path)

        # warning
        is_warning_triggered = False
        max_gap = timedelta(seconds=0.5)
        for id in object_bboxes.keys():  # 遍历所有被检测到的对象的包围框
            if id not in self.state:            # 如果对象ID不在已报警的状态记录中
                if id not in self.det_record or now - self.det_record[id] > max_gap:  # 如果没有检测记录或者距离上次检测到现在的时间间隔超过了设定的最大间隔
                    # 重置计数器为1并记录当前时间
                    self.counter[id] = 1
                    self.det_record[id] = now
                
                elif now - self.det_record[id] <= max_gap: # 如果距离上次检测到现在的时间间隔小于等于设定的最大间隔
                    # 增加对应ID的计数器
                    self.counter[id] += 1
                    self.det_record[id] = now
                    # 如果连续检测到的次数超过了10次，则触发报警
                    if self.counter[id] > 10:  # 连续检测到了k次进行报警
                        is_warning_triggered = True
                        # 记录日志信息
                        logger.info(f"{self.pname} warning! id = {id}", "正在交给ried_task4进行计算")
                        # 保存图片并获取路径和图片本身
                        warningImage_path, warningImage = self.on_save_img(object_bboxes[id], id, self.config.warning_path)
                        # 将ID添加到已报警的状态记录中
                        self.state.append(id)
                        self.counter[id] = 0
                        
                        # 定义临时警告图片存储路径
                        temporary_warning_path = "res/images/reid_tmp_data/phone_warning"
                        if not os.path.exists(temporary_warning_path):
                            os.makedirs(temporary_warning_path)

                        # 获取原始图片文件名
                        warningImage_filename = os.path.basename(warningImage_path)
                        # 构建目标文件路径（包括文件名）
                        destination_file_path = os.path.join(temporary_warning_path, warningImage_filename)

                        # 把报警图片复制到目标文件路径
                        shutil.copy(warningImage_path, destination_file_path)
                        print("调试249")
                        # 发送警告信息！！！！！！！！！！！！！！！！！！！重复报警，后续可以删除
                        WarnKit.send_warn_result(self.pname, self.output_dir, self.stream_cam_id, 1, 1,
                                                warningImage, self.config.stream_export_img_enable,
                                                self.config.stream_web_enable)
        if is_warning_triggered:
            # 构建请求数据
            print("调试256")
            data = {
                "query_directory_or_id":  destination_file_path,
                "gallery_directory": "res/images/reid_tmp_data/id_gt"
            }
            # 发送POST请求
            response = requests.post("http://localhost:5000/process2", json=data, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                print("zzzzzzzzzzzzzzzzzzzzyyyyyyyyyyyyyyyyyyyyyyyyy POST successfully")
            else:
                print("zzzzzzzzzzzzzzzzzzzzyyyyyyyyyyyyyyyyyyyyyyyyy POST failed, status code:", response.status_code)
                    
                
                
                
                
    #####################################################################################################

    def copy_images_to_temporary_directory(self):
        original_warning_path = self.config.warning_path
        temporary_warning_path = "res/images/reid_tmp_data/phone_warning"

        # 确保临时目录存在，如果不存在则创建它
        if not os.path.exists(temporary_warning_path):
            os.makedirs(temporary_warning_path)

        # 获取原始警告目录中的最新5个文件
        warning_files = sorted(os.listdir(original_warning_path), key=lambda x: os.path.getmtime(os.path.join(original_warning_path, x)), reverse=True)[:5]
        
        # 遍历文件，并将它们复制到新的临时目录
        for file_name in warning_files:
            # 构建原始文件的完整路径
            original_file_path = os.path.join(original_warning_path, file_name)
            # 构建新的目标文件路径
            temporary_file_path = os.path.join(temporary_warning_path, file_name)
            # 复制文件
            shutil.copy(original_file_path, temporary_file_path)


    def on_save_img(self, bbox, id, path):
        # 检查路径是否存在
        if not os.path.exists(path):
            # 如果路径不存在，创建路径
            os.makedirs(path)
        # 接下来是你的图像处理逻辑
        img = self.frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
        # 确保文件名是路径的一部分
        image_path = os.path.join(path, f"0_{self.stream_cam_id}_{time_str}_0.jpg")
        cv2.imwrite(image_path, img)
        return image_path, img
    
    def count_images_in_directory(self, directory):
        # 返回目录中的jpg图片数量
        return len(glob.glob(os.path.join(directory, '*.jpg')))

    def clear_directory(self, directory):
        # 删除目录中的所有文件
        files = glob.glob(os.path.join(directory, '*'))
        for f in files:
            os.remove(f)


def create_process(shared_data, config_path: str):
    phoneComp: PhoneComponent = PhoneComponent(shared_data, config_path)  # 创建组件
    phoneComp.start()  # 初始化
    phoneComp.update()  # 算法逻辑循环





if __name__ == "__main__":
    time_str = time.strftime('%Y%m%d', time.localtime())
    print(time_str)
    id_str = f"{2:02d}"
    print(id_str)



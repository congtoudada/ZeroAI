from collections import deque

import cv2
import numpy as np


class VadItem:
    def __init__(self):
        self.cam_id = 0  # 摄像头id
        self.capacity = 7  # 队列容量
        self.queue: deque = None  # 帧缓存队列
        self.frame_id = 0  # 帧push_id
        # 帧异常检测
        self.last_frame_id = -100  # 上次执行帧异常检测的时间
        self.frame_score = 0  # 最近一次帧异常报警分数
        self.frame_valid = 0  # 连续异常帧计数
        # 对象级异常检测
        self.last_obj_id = -100  # 上次执行对象异常检测的时间
        self.obj_score = 0  # 最近一次对象级异常报警分数
        self.det_queue: deque = None  # bbox缓存队列

    def init(self, cam_id, capacity):
        self.cam_id = cam_id
        self.capacity = capacity
        self.queue = deque(maxlen=capacity)
        self.frame_id = 0
        # 帧异常检测
        self.last_frame_id = -1
        self.frame_score = 0
        self.frame_valid = 0
        # 对象级异常检测
        self.last_obj_id = -1
        self.obj_score = 0
        self.det_queue = deque(maxlen=capacity)

    def push(self, frame, frame_id, input_det=None):
        if frame_id - self.frame_id > 1:
            print("VAD override, will lost frame {}".format(frame_id - self.frame_id))
        self.queue.append(frame)  # 满了自动弹出队头
        self.det_queue.append(input_det)
        self.frame_id = frame_id
        if self.last_frame_id == -1:  # 丢弃掉第一轮结果
            self.last_frame_id = frame_id
            self.last_obj_id = frame_id
        # print("len:", len(self.queue))

    def get_batch(self, num, scale_hw=None):
        if len(self.queue) < num:
            return None
        batch = list(self.queue)[-num:]
        if scale_hw is not None:
            for i in range(len(batch)):
                batch[i] = cv2.resize(batch[i], scale_hw)
        return np.array(batch)

    def update_frame_score(self, score, threshold, times, valid_limit):
        self.frame_score = score
        if score >= threshold:
            self.frame_valid += times
            if self.frame_valid > valid_limit * 2:
                self.frame_valid = valid_limit * 2
        else:
            self.frame_valid -= 1
            if self.frame_valid < 0:
                self.frame_valid = 0



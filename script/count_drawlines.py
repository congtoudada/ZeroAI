"""
计数绘制线工具: q退出,s保存extract_frames.py
"""
import os.path
import signal

import cv2
import numpy as np

from zero.utility.config_kit import ConfigKit
from zero.utility.yaml_kit import YamlKit

# 用户设置变量
cap = cv2.VideoCapture("res/videos/renlian/renlian1.mp4")  # 打开本地视频
expect_preview = (1280, 720)  # 期望分辨率
frame_index = 50  # 替换为你想要读取的帧索引
input_file = "conf/algorithm/business/count/count_renlian.yaml"  # 与output不一致，每次可以在结果集基础上预览
output_file = "script/output/count/drawn_points.yaml"  # 输出路径
auto_save = True
mode_str = ['red', 'green']
mode_idx = 1  # 0:red 1:green

# 预定义变量，用于保存上一次点击的位置和所有绘制的点的位置
prev_point = None
arrow_length = 20
arrow_angle = 30  # 箭头夹角
drawn_points = []  # 用于保存所有绘制的点的位置


# 计算两点之间的距离
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


# 计算两点之间的角度
def calculate_angle(point1, point2):
    return np.arctan2(point2[1] - point1[1], point2[0] - point1[0])


# 绘制线条和箭头
def draw_line_with_arrow(start_point, end_point, color):
    cv2.line(frame, start_point, end_point, color, 2)  # 绘制线条
    distance = calculate_distance(start_point, end_point)
    if distance >= arrow_length:
        angle = calculate_angle(start_point, end_point)
        # 计算箭头末端的位置
        arrow_end1 = (int(end_point[0] - arrow_length * np.cos(angle + np.radians(arrow_angle))),
                      int(end_point[1] - arrow_length * np.sin(angle + np.radians(arrow_angle))))
        arrow_end2 = (int(end_point[0] - arrow_length * np.cos(angle - np.radians(arrow_angle))),
                      int(end_point[1] - arrow_length * np.sin(angle - np.radians(arrow_angle))))
        # 绘制箭头
        cv2.line(frame, end_point, arrow_end1, color, 2)
        cv2.line(frame, end_point, arrow_end2, color, 2)


# 重新绘制图像
def redraw_frame():
    global frame, prev_point
    # 重新读取原始帧以清除先前的绘制
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    frame = cv2.resize(frame, expect_preview, interpolation=cv2.INTER_LINEAR)
    frame = cv2.rectangle(frame, (dead_zone, dead_zone), (frame.shape[1] - dead_zone, frame.shape[0] - dead_zone),
                          (0, 0, 187), 2)
    pre_process(frame, input_file)
    # 重新绘制所有剩余的点和线条
    for i, point in enumerate(drawn_points):
        cv2.circle(frame, point, 3, (0, 255, 0), -1)  # 绘制提示圆圈
        if i > 0:  # 绘制线条和箭头
            draw_line_with_arrow(drawn_points[i - 1], point, (0, 0, 255))

    # 重置 prev_point 以防止绘制断开的线条
    if drawn_points.__len__() > 0:
        prev_point = (drawn_points[drawn_points.__len__() - 1][0], drawn_points[drawn_points.__len__() - 1][1])
    else:
        prev_point = None
    cv2.imshow("Frame", frame)


# 鼠标点击事件的回调函数
def draw_line_with_fixed_arrow(event, x, y, flags, param):
    global prev_point, drawn_points
    # 自动吸附到图像边缘
    if x < dead_zone:
        x = 0
    if y < dead_zone:
        y = 0
    if x > frame.shape[1] - dead_zone:
        x = frame.shape[1]
    if y > frame.shape[0] - dead_zone:
        y = frame.shape[0]
    if event == cv2.EVENT_LBUTTONDOWN:  # 当左键按下时
        if prev_point is not None:
            # cv2.line(frame, prev_point, (x, y), (0, 0, 255), 2)  # 绘制线条
            draw_line_with_arrow(prev_point, (x, y), (171, 218, 136))  # 绘制箭头
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # 绘制提示圆圈
            cv2.imshow("Frame", frame)
        else:
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # 绘制提示圆圈
            cv2.imshow("Frame", frame)
        # print((x, y))
        drawn_points.append((x, y))  # 记录绘制的点的位置
        prev_point = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN and drawn_points:  # 当右键按下时
        drawn_points.pop()  # 移除最后一个点
        redraw_frame()  # 重新绘制图像


def pre_process(frame, input_file):
    if os.path.exists(input_file):
        # local_dict: dict = YamlKit.read_yaml(input_file)
        local_dict: dict = ConfigKit.load(input_file)
        if not local_dict.__contains__("count"):
            return
        if local_dict['count'].__contains__('red'):
            pre_process_draw(frame, local_dict['count']['red'], (0, 0, 255))
        if local_dict['count'].__contains__('green'):
            pre_process_draw(frame, local_dict['count']['green'], (0, 255, 0))


def pre_process_draw(frame, points, color):
    local_prev_point = None
    for point_str in points:
        point = (int(float(point_str.split(',')[0]) * expect_preview[0]), int(float(point_str.split(',')[1]) * expect_preview[1]))
        if local_prev_point is not None:
            # cv2.line(frame, local_prev_point, (point[0], point[1]), (0, 0, 0), 2)  # 绘制线条
            draw_line_with_arrow(local_prev_point, (point[0], point[1]), color)  # 绘制箭头
            cv2.circle(frame, (point[0], point[1]), 3, (0, 0, 0), -1)  # 绘制提示圆圈
            local_prev_point = (point[0], point[1])
        else:
            cv2.circle(frame, (point[0], point[1]), 3, (0, 0, 0), -1)  # 绘制提示圆圈
            local_prev_point = (point[0], point[1])


# 保存按钮点击事件的回调函数
def save_points():
    global drawn_points
    # 将绘制的点位置写入文本文件
    result = []
    for point in drawn_points:
        result.append(f"{1.0 * point[0]/expect_preview[0]:.3f},{1.0 * point[1]/expect_preview[1]:.3f}")
    YamlKit.write_yaml(output_file, {
        'count': {mode_str[mode_idx]: result}
    })
    print(f"Drawn points saved to {output_file}")


def handle_termination(signal_num, frame):
    print(f'接收到信号 {signal_num}, 准备退出...')
    if auto_save:
        save_points()


# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Failed to open videos.")
    exit()

# 获取视频帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 读取指定帧
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
frame = cv2.resize(frame, expect_preview, interpolation=cv2.INTER_LINEAR)

if not ret:
    print(f"Error: Failed to read frame {frame_index}.")
    exit()

dead_zone = 20
frame = cv2.rectangle(frame, (dead_zone, dead_zone), (frame.shape[1] - dead_zone, frame.shape[0] - dead_zone),
                      (0, 0, 187), 2)
pre_process(frame, input_file)

# 创建窗口并设置鼠标回调函数
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_line_with_fixed_arrow)

# 在窗口中显示图像
cv2.imshow("Frame", frame)

signal.signal(signal.SIGINT, handle_termination)
signal.signal(signal.SIGTERM, handle_termination)

while True:
    key = cv2.waitKey(1) & 0xFF
    # 检测是否按下 's' 键（保存按钮）
    if key == ord('s'):
        save_points()
    # 检测是否按下 'q' 键（退出按钮）
    elif key == ord('q'):
        break
# 关闭窗口
cv2.destroyAllWindows()

# 释放视频对象
cap.release()
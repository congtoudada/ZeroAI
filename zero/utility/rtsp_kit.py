# 推流工具类
import cv2
import mediapipe as mp
import numpy as np
import subprocess as sp


class RtspKit:
    def __init__(self, rtsp_url, width, height, fps):
        # ffmpeg command 保存进程参数
        self.command = [
            'ffmpeg',
            # 're',#
            # '-y', # 无需询问即可覆盖输出文件
            '-f', 'rawvideo',  # 强制输入或输出文件格式
            '-vcodec', 'rawvideo',  # 设置视频编解码器。这是-codec:v的别名
            '-pix_fmt', 'bgr24',  # 设置像素格式
            '-s', "{}x{}".format(width, height),  # 设置图像大小
            '-r', str(fps),  # 设置帧率
            '-i', '-',  # 输入
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'rtsp',  # 强制输入或输出文件格式
            rtsp_url]
        print("初始化rtsp推流器: " + rtsp_url)
        self.rtsp_proxy = sp.Popen(self.command, stdin=sp.PIPE)

    def push(self, frame):
        if frame is not None:
            print("rtsp push!")
            self.rtsp_proxy.stdin.write(frame.tostring())
        else:
            print("push rtsp is failed!")

    def __del__(self):
        if self.rtsp_proxy is not None:
            self.rtsp_proxy.stdin.close()  # 首先关闭stdin
            self.rtsp_proxy.kill()
            self.rtsp_proxy.wait()
            self.rtsp_proxy = None  # 设置为None以避免再次调用kill或wait

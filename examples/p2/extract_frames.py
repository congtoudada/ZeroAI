"""
抽帧脚本
"""
import cv2
import numpy as np
import os
from pathlib import *
import argparse

phase = "training"
video_path = rf"./dataset/yuanxing/keliu2/videos/"
# video_path = r"E:/Practice/AI/Demo/jiuyuan/dataset/test/videos/"
img_path = rf"./dataset/yuanxing/keliu2/frames/"
# img_path = r"E:/Practice/AI/Demo/jiuyuan/dataset/test/frames/"
step = 5  # 每step帧采样一次

# Config
# def get_configs():
#     parser = argparse.ArgumentParser(description="Extract_frames")
#     parser.add_argument("--dataset", type=str, default="custom")
#     parser.add_argument("--phase", type=str, default='test', choices=['train', 'test'])
#     args = parser.parse_args()
#
#     return args

def extract():
    path_videos = f"{video_path}"
    path_frames = f"{img_path}"

    films = list()
    print(Path(path_videos).name)
    files = (x for x in Path(path_videos).iterdir() if x.is_file())
    for file in files:
        print(str(file.name).split(".")[0], "is a file!")
        films.append(file)

    video_count = films.__len__()
    print(f"总视频数: {video_count}")
    for i, film in enumerate(films):
        count = 0
        vidcap = cv2.VideoCapture(str(film))
        success, image = vidcap.read()
        mapp = str(film.name).split(".")[0]
        step_index = 0
        while success:
            # name = f"{path_frames}{mapp}/{count}.jpg"
            # name = "{}{}/{:03d}.jpg".format(path_frames, mapp, count) # "{:03d}.jpg".format(frames)
            # name = "{}{}/{}_{}.jpg".format(path_frames, mapp, mapp, count)  # "{:03d}.jpg".format(frames)
            name = "{}{}/{}_{}.jpg".format(path_frames, mapp, mapp, count)  # "{:03d}.jpg".format(frames)
            if not os.path.isdir(f"{path_frames}{mapp}"):
                print(f"{path_frames}{mapp}")
                os.mkdir(f"{path_frames}{mapp}")
            if step_index >= step:
                step_index = 0
                cv2.imwrite(name, image)  # save frames as JPEG file
            else:
                step_index += 1
            success, image = vidcap.read()
            count += 1

        video_count -= 1
        print(f"当前进度：{(films.__len__() - video_count) / films.__len__() * 100 }%")


if __name__ == '__main__':
    # args = get_configs()
    extract()
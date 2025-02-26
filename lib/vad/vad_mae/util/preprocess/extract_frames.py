"""
抽帧脚本
"""
import cv2
import numpy as np
import os
from pathlib import *
import argparse


def extract(video_path, img_path, step):
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
            name = "{}/{}/{}.png".format(path_frames, mapp, count)  # "{:03d}.jpg".format(frames)
            if not os.path.isdir(f"{path_frames}/{mapp}"):
                print(f"{path_frames}/{mapp}")
                os.makedirs(f"{path_frames}/{mapp}", exist_ok=True)
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
    step = 0
    # 抽帧Avenue、ShanghaiTech
    # phase = "test"
    # video_path = rf"H:\AI\dataset\VAD\Avenue\Avenue_Dataset\Avenue Dataset\{phase}ing_videos"
    # img_path = rf"H:\AI\dataset\VAD\Avenue\Avenue_Dataset\Avenue Dataset\{phase}\frames"
    # extract(video_path, img_path, step)
    # phase = "train"
    # video_path = rf"H:\AI\dataset\VAD\Avenue\Avenue_Dataset\Avenue Dataset\{phase}ing_videos"
    # img_path = rf"H:\AI\dataset\VAD\Avenue\Avenue_Dataset\Avenue Dataset\{phase}\frames"
    # extract(video_path, img_path, step)

    phase = "test"
    video_path = rf"H:\AI\dataset\VAD\Avenue\Avenue_Dataset\Avenue Dataset\{phase}ing_videos"
    img_path = rf"H:\AI\dataset\VAD\Avenue\Avenue_Dataset\Avenue Dataset\{phase}\frames"
    extract(video_path, img_path, step)
    phase = "train"
    video_path = rf"H:\AI\dataset\VAD\Avenue\Avenue_Dataset\Avenue Dataset\{phase}ing_videos"
    img_path = rf"H:\AI\dataset\VAD\Avenue\Avenue_Dataset\Avenue Dataset\{phase}\frames"
    extract(video_path, img_path, step)

    # # 抽帧UBnormal
    # img_path = rf"H:\AI\dataset\VAD\UBnormal\train\frames"
    # for i in range(5):
    #     video_path = rf"H:\AI\dataset\VAD\UBnormal\Scene{i+1}"
    #     extract(video_path, img_path, step)


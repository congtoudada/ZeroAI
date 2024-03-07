#!/usr/bin/env bash

# 测试图片文件
image_dir="data/test_image" # 测试图片的目录
out_dir="output/"  # 保存检测结果
python face_search.py --image_dir $image_dir

# 测试视频文件
video_file="data/test-video.mp4"
python face_search.py --video_file $video_file

# 测试摄像头
video_file="0"
python face_search.py --video_file $video_file


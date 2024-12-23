import cv2
from cv2 import VideoWriter


class SaveVideoHelper:
    def __init__(self, output_path, enable_resize, width, height, fps=24):
        self.output_path = None
        self.vid_writer: VideoWriter = None
        self.fps = fps
        self.enable_resize = enable_resize
        self.width = width
        self.height = height
        self.set_output(output_path, fps, width, height)

    def set_output(self, output_path, fps, width, height):
        if self.vid_writer is not None:
            self.vid_writer.release()
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        if output_path is not None:
            self.vid_writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
            )

    def write(self, frame):
        if self.vid_writer and frame is not None:
            if self.enable_resize:
                self.vid_writer.write(cv2.resize(frame, (self.width, self.height)))  # 确保长宽与预定义一致（由内部维护）
            else:
                self.vid_writer.write(frame)  # 确保长宽与预定义一致（由外部维护）

    def on_destroy(self):
        if self.vid_writer is not None:
            self.vid_writer.release()


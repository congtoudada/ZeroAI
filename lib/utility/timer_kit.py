import time


class TimerKit(object):
    """A simple timer."""
    def __init__(self, max_flag=10):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.  # 平均耗时
        self.max_time = 0.  # 峰值耗时
        self.recent_time = 0.  # 最近一次耗时
        self.duration = 0.
        self.MAX_FLAG = max_flag  # 前10次不记录

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.calls += 1
        self.diff = time.time() - self.start_time
        self.recent_time = self.diff
        if self.calls < self.MAX_FLAG:
            return
        self.total_time += self.diff
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        if self.diff > self.max_time:
            self.max_time = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.max_time = 0.
        self.recent_time = 0.
        self.duration = 0.

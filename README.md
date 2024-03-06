# ZeroAI

# 算法层

## 共享内存（更新中）

```python
class SharedKey(Enum):
    """
    进程间共享数据的Key常量
    """
    """
    全局
    """
    EVENT_ESC = 0  # 退出事件
    """
    视频流
    """
    STREAM_WAIT_COUNTER = 100  # 视频流初始化等待计时器（初始化用）
    STREAM_LOCK = 101  # 锁（必要时可用）
    STREAM_ORIGINAL_FRAME = 102  # 读取的原始视频图像
    STREAM_ORIGINAL_WIDTH = 103  # 原始视频图像宽
    STREAM_ORIGINAL_HEIGHT = 104  # 原始视频图像高
    STREAM_ORIGINAL_CHANNEL = 105  # 原始视频图像通道数
    STREAM_ORIGINAL_FPS = 106  # 原始视频图像帧率

```
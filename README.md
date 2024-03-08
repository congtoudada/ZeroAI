# ZeroAI（持续更新中）

ZeroAI是一个由**配置文件驱动**的**组件式**，基于**视频流**的**多进程**AI框架

## 目录结构

```
.
├── bin                    # main脚本，运行时内容
├── conf                   # 各种配置
│   ├── algorithm              # 算法配置
│   ├── cam                    # 视频流配置（一个视频流配置可对应多个算法）
│   ├── global                 # 全局配置
│   ├── application-dev.yaml   # 项目根配置（开发环境）
│   └── application-pro.yaml   # 项目根配置（生产环境）
├── lib                    # 各类算法
│   ├── business               # 业务类算法
│   ├── detection              # 目标检测算法
│   ├── face                   # 人脸识别算法
│   ├── mot                    # 多目标追踪算法
│   └── reid                   # 重识别算法
├── log                    # 日志（自动生成）
├── pretrained             # 预训练权重
├── res                    # 资源
│   ├── images                 # 图像资源
│   └── videos                 # 视频资源
├── script                 # 脚本工具
├── zero                   # ZeroAI框架
│   ├── core                   # 框架核心代码
│   └── utility                # 框架工具脚本
├── README.md              # 说明文档
├── setup.py               # 自定义包安装脚本
└── requirements.txt       # 项目依赖文件
```

## 安装





## 运行





## TODO

* 拓展算法
* 接入Web后端
* 效果演示

## 关键概念

### Component 分类

* **基础组件**：最核心最基础的组件。
  * LauncherComponent：启动组件。负责管理整个框架的初始化与运作
  * StreamComponent：流组件。一个摄像头对应一个流组件，负责取流并初始化各个算法
* **算法组件**：单独的进程运行，所有算法都是基于流的，因此一个流组件可装配若干算法组件进行推理（适用于实时算法）
* **服务组件**：单独的进程运行，所有摄像头进程共享，处理并响应所有摄像头的请求（适用于非实时算法，如人脸识别进程等）
* **帮助组件**：需要依赖其他组件运行，拓展某些组件的功能（适用于为算法添加功能，如为计数算法添加人脸识别）

### Component 生命周期函数

* `__init__`：构造函数。主要用于声明变量或进行简单初始化
* `on_start`：初始化时调用一次。主要用于复杂的初始化工作
* `on_update`：每帧调用，具体频率由配置文件指定。主要用于运行算法逻辑
* `on_destroy`：销毁时调用。释放资源
* `on_analysis`：每帧调用，具体频率由配置文件指定。自动打印日志

> 某些组件可能有自己单独的生命周期函数

框架流程纵向图

![第一阶段纵向图](README.assets/第一阶段纵向图.jpg)

> 每个Component通常对应一个Info，用于挂载配置参数

框架流程横向图

![第一阶段横向图](README.assets/第一阶段横向图.jpg)

### Config 配置

参考SpringBoot配置规则，编写自定义解析模块：

* 支持`INCLUDE`关键字，通过加载其他配置文件来为自身添加所需配置；
* 基于YAML文件，支持配置细粒度重载；充分利用python动态语言特性，支持通过`set_attrs`自动赋值。

工作流程：编写yaml配置文件 --> 编写对应的info脚本 --> 利用ConfigKit工具加载配置 --> 借助info脚本使用配置

### 共享内存

基于`multiprocessing.Manager().dict()`（按照普通dict使用即可）实现进程间通信，进程安全

* `shared_data`：摄像头独享，用于单个视频流内算法的通信
* `global_shared_data`：全局共享内存，所有摄像头共享，利用`camera_id`可以访问到特定摄像头的`shared_data`

### 共享 Key

共享内存基于`dict`，因此需要有Key，为了避免繁琐的字符串拼写，所有Key使用枚举管理

* SharedKey：存放全局Key和视频流进程所需Key
* FaceKey：存放人脸识别相关Key（避免SharedKey内的Key过多）

#### SharedKey

```python
class SharedKey(Enum):
    """
    单个摄像头进程内，摄像头与算法共享数据的Key常量
    """
    """
    全局
    """
    EVENT_ESC = 0  # 退出事件
    WAIT_COUNTER = 1  # 服务初始化等待计数器（取决于服务数量，初始化用）
    CAMERAS = 2  # 相机dict列表
    # ------------------ 以下Key对global_shared_data无效 ------------------
    """
    视频流
    """
    STREAM_GLOBAL = 100  # 全局共享内存的引用
    STREAM_WAIT_COUNTER = 101  # 视频流初始化等待计数器（取决于算法数量，初始化用）
    # ---
    STREAM_FRAME_INFO = 102  # 视频流信息 (package)
    STREAM_FRAME_ID = 103  # 原始图像ID（每次成功读取新的FRAME都会更新ID，避免算法重复处理相同帧）
    STREAM_FRAME = 104  # 原始图像
    # ---
    STREAM_ORIGINAL_WIDTH = 105  # 原始图像宽
    STREAM_ORIGINAL_HEIGHT = 106  # 原始图像高
    STREAM_ORIGINAL_CHANNEL = 107  # 原始图像通道数
    STREAM_ORIGINAL_FPS = 108  # 原始视频图像帧率
    STREAM_URL = 109  # 摄像头取流地址
    STREAM_CAMERA_ID = 110  # 摄像头id
    STREAM_UPDATE_FPS = 111  # 算法最小update间隔
    """
    目标检测
    """
    # ---
    DETECTION_INFO = 200  # 检测算法信息 (package)
    DETECTION_ID = 201  # 当前帧ID（每次成功读取新的FRAME都会更新ID，避免算法重复处理相同帧）
    DETECTION_FRAME = 202  # 读取的检测图像
    # 算法输出结果 shape: [n, 6]
    # n: n个对象
    # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
    #   [0]: x1
    #   [1]: y1
    #   [2]: x2
    #   [3]: y2
    # [4]: 置信度
    # [5]: 类别 (下标从0开始)
    DETECTION_OUTPUT = 203
    # ---
    DETECTION_TEST_SIZE = 204  # 目标检测输入尺寸 (暂时没用)
    """
    多目标追踪
    """
    # ---
    MOT_INFO = 300  # 多目标追踪算法信息 (package)
    MOT_ID = 301  # 当前帧ID（每次成功读取新的FRAME都会更新ID，避免算法重复处理相同帧）
    MOT_FRAME = 302  # 读取的检测图像
    # 算法输出结果 shape: [n, 7]
    # n: n个对象
    # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
    #   [0]: x1
    #   [1]: y1
    #   [2]: x2
    #   [3]: y2
    # [4]: 置信度
    # [5]: 类别 (下标从0开始)
    # [6]: id
    MOT_OUTPUT = 303
    # ---
```

#### FaceKey

```python
class FaceKey(Enum):
    """
    人脸识别Key
    """
    # --- 实际请求key REQ + port ---
    REQ = 0
    REQ_CAM_ID = 1
    REQ_PID = 2  #
    REQ_OBJ_ID = 4  # 追踪对象的id
    REQ_IMAGE = 5  # 人脸图像
    # --- 实际响应key RSP + port + pid ---
    RSP = 10
    RSP_OBJ_ID = 11
    RSP_PER_ID = 12  # 人脸识别结果（1为陌生人）
    RSP_SCORE = 13  # 人脸识别分数
```


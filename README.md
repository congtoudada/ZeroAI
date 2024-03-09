# ZeroAI（持续更新中）

ZeroAI是一个由**配置文件驱动**的**组件式**，基于**视频流**的**多进程**AI框架，支持任意数量AI算法的快速部署与整合。

* 高效：

  * 基于Python多进程编写，相比于Socket通信，效率更高
  * 基于GPU的推理，可采用Tensor RT或ONNX-GPU加速

* 灵活性强：

  * 基于配置文件驱动，通过修改配置文件，可以实现算法的自由切换与高度自定义配置

  * 基于Port的设计，使得输出结果可以复用，同时单个算法也可以处理多输入

* 可扩展性强：

  * 基于组件式的设计，无需改动框架结构，可以轻松实现逻辑的横向、纵向扩展

* 易上手：

  * 框架预提供了丰富的组件，可以很轻松地根据需要接入自己的业务
  * 配备贴心的教程文档，助力开发人员快速上手

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
├── output                 # 框架输出
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



## 演示



## 自定义



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

```python
class Component(ABC):
    def __init__(self, shared_data: dict):
        self.pname = "component"
        self.enable = True
        self.shared_data: dict = shared_data
        self.config: BaseInfo = None
        self.pname = f"[ {os.getpid()}:component ]"
        self.esc_event = None

    def on_start(self):
        if self.shared_data is not None:
            self.esc_event = self.shared_data[SharedKey.EVENT_ESC]
        if LogKit.load_info(self.config):  # 新进程设置日志
            pass
            # logger.info(f"{self.pname} 成功运行日志模块! 输出路径: {self.config.log_output_path}")
        else:
            logger.info(f"{self.pname} 日志模块被关闭!")

    def on_update(self) -> bool:
        return True

    def on_destroy(self):
        logger.info(f"{self.pname} destroy!")

    def on_analysis(self):
        pass

    ...
```

> Tips：某些组件可能有自己单独的生命周期函数

框架流程纵向图

![第一阶段纵向图](README.assets/第一阶段纵向图.jpg)

框架流程横向图

![第一阶段横向图](README.assets/第一阶段横向图.jpg)

> Tips：
>
> * 每个Component通常对应一个Info，用于存储配置参数。
> * Basedxxx，通常说明组件的输入来自哪里；Basexxx，通常说明组件输出到哪里。

### Info 文件

通常每一个Component都可以编写一个对应的Info文件，用于存放从配置文件中读取的配置信息，共享内存的Key等。使用Info文件可以有效避免手动输入字符串导致的错拼、漏拼的问题。

```python
class AppInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.cam_list = []
        self.service = []
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
```

> Tips：切记，除非你明确知道后果，如果需要加载配置文件的内容，需将`super().__init__(data)`放在变量声明的最后。

### Config 配置

参考SpringBoot配置规则，编写自定义解析模块：

* 支持`INCLUDE`关键字，通过加载其他配置文件来为自身添加所需配置；
* 基于YAML文件，支持配置细粒度重载；充分利用python动态语言特性，支持通过`set_attrs`自动赋值。

工作流程：编写yaml配置文件 --> 编写对应的info脚本 --> 利用ConfigKit工具加载配置 --> 借助info脚本使用配置

Yolox yaml配置参考：

```yaml
INCLUDE:
  - conf/algorithm/detection/yolox/yolox_root.yaml
stream:
  input_ports:  # 输入端口 eg.SharedKey.STREAM_FRAME_INFO-camera1
    - camera1
    - camera2
detection:
  output_port: yolox  # 输出端口 eg.SharedKey.STREAM_FRAME_INFO-yolox-camera1
yolox:
  args:
    expn: head  # 实验名称
    path: null  # 取流路径（为None则通过内部框架取流）
    save_result: False # 是否存储视频
    ...
```

YoloxInfo参考：

在最后调用`super().__init__(data)`来重载配置

```python
class YoloxInfo(DetInfo):
    def __init__(self, data: dict = None):
        self.yolox_vis = False  # 是否使用opencv可视化（测试用）
        self.yolox_args_expn = ""  # 实验名称
        self.yolox_args_path = None  # 取流路径（为None则通过内部框架取流）
        self.yolox_args_save_result = False  # 是否存储视频
        ...
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
```

YoloxComponent参考：

其中`self.config`就是根据路径从配置文件中加载的配置

```python
class YoloxComponent(BaseDetComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: YoloxInfo = YoloxInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:yolox for {self.config.yolox_args_expn}]"
        # 自身定义
        self.output_dir = ""  # 输出目录
        ...
```

### 共享内存（可选）

基于`multiprocessing.Manager().dict()`（按照普通dict使用即可）实现进程间通信，进程安全

* `shared_data`：摄像头独享，用于单个视频流内算法的通信
* `global_shared_data`：全局共享内存，所有摄像头共享，利用`camera_id`可以访问到特定摄像头的`shared_data`

### 共享内存Key（可选）

共享内存基于`dict`，因此需要有Key，为了避免繁琐的字符串拼写，所有Key使用枚举管理

* SharedKey：存放全局Key和视频流进程所需Key
* FaceKey：存放人脸识别相关Key（避免SharedKey内的Key过多）

#### SharedKey参考

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
    LOCK = 3  # 锁
    
    ...
    
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
    ...
```

#### FaceKey参考

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

## TODO

* 找工作（游戏开发）
* 拓展算法
* 接入Web后端
* 效果演示
* 支持Tensor RT

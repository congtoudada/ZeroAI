# ZeroAI

ZeroAI是一个由**配置文件驱动**的**组件式**，基于**视频流**的**多进程**AI框架，支持不同种类AI算法的快速部署与整合。

# 重要目录/文件

```sh
.
├── bin                    # main脚本存放位置
├── conf                   # 配置
├── document               # 文档
├── examples               # 示例
├── lib                    # 库
│   ├── detection              # 目标检测
│   ├── facr                   # 人脸识别
│   ├── mot                    # 多目标追踪
│   ├── network                # 网络相关
│   ├── reid                   # 行人重识别
│   ├── utility                # 工具库
│   └── zero                   # 框架核心库
├── install.py             # 自定义包安装器
└── requirements.txt       # 项目依赖文件
```

# 更新日志

## v1.0 Detection

ZeroAI v1.0-Detection，包含yolox，详见[示例P2：目标检测](./document/示例P2：目标检测)

## v1.0 Pure

ZeroAI v1.0-Pure，只包含ZeroAI最核心的代码，即只提供取流接口，没有集成任何服务和算法，适合入门学习，熟悉ZeroAI框架核心思想，详见[示例P1：打印视频流](./document/示例P1：打印视频流)

# ZeroAI（持续更新中）

ZeroAI是一个由**配置文件驱动**的**组件式**，基于**视频流**的**多进程**AI框架

目录结构

```
.
├── bin                    # main脚本，运行时内容
├── conf                   # 各种配置
│   ├── algorithm              # 算法配置
│   ├── cam                    # 视频流配置（一个视频流配置可对应多个算法）
│   ├── log                    # 日志配置
│   ├── application-dev.yaml   # 项目根配置（开发环境）
│   └── application-pro.yaml   # 项目根配置（生产环境）
├── lib                    # 各类算法
│   ├── business               # 业务类算法
│   ├── detection              # 目标检测算法
│   ├── face                   # 人脸识别算法
│   ├── mot                    # 多目标追踪算法
│   └── reid                   # 重识别算法
├── log                    # 日志
├── pretrained             # 预训练权重
├── res                    # 资源
│   ├── images                 # 图像资源
│   └── videos                 # 视频资源
├── script                 # 脚本工具
├── zero                   # ZeroAI框架
├── README.md              # 说明文档
├── setup.py               # 自定义包安装脚本
└── requirements.txt       # 项目依赖文件
```

# 算法层

## 共享内存（更新中）


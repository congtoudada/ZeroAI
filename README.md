# ZeroAI

ZeroAI是一个由**配置文件驱动**的**组件式**，基于**视频流**的**多进程**AI框架，支持不同种类AI算法的快速部署与整合。

# 内容目录

```sh
.
├── bin                    # main脚本，项目入口
├── conf                   # 配置
├── examples               # 示例目录
├── lib                    # 库
│   ├── utility                # 工具库
│   └── zero                   # 框架核心库
├── res                    # 资源
└── requirements.txt       # 项目依赖文件
```

# 更新日志

## v1.0 Pure

ZeroAI v1.0-Pure，只包含ZeroAI最核心的代码，即只提供取流接口，没有集成任何服务和算法，适合入门学习，熟悉ZeroAI框架流程（入门示例请参考[示例P1：打印视频流](./document/示例P1：打印视频流)）

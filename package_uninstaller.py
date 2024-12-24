import os
import subprocess

# 定义要搜索的目录
uninstall_package = ['zero', 'yolox', 'bytetrack', 'insight', 'simple_http', 'business']  # 修改为你的工程目录

for i, name in enumerate(uninstall_package):
    print(f"Running 'pip uninstall {name} -y'")
    # 在 setup.py 所在的目录下执行 python setup.py develop
    subprocess.run(['pip', 'uninstall', name, '-y'])

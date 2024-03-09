import os
import subprocess

# 定义要搜索的目录
search_dir = './'  # 修改为你的工程目录

# 递归搜索目录下的 setup.py 文件
for root, dirs, files in os.walk(search_dir):
    if 'setup.py' in files:
        # 构造完整的 setup.py 文件路径
        setup_py_path = os.path.join(root, 'setup.py')
        print(f"Running 'python {setup_py_path} develop' in {root}...")
        # 在 setup.py 所在的目录下执行 python setup.py develop
        subprocess.run(['python', os.path.basename(setup_py_path), 'develop'], cwd=root, check=True)
        print(f"Finished running 'python {setup_py_path} develop' in {root}\n")

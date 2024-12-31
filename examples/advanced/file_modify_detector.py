import os
import time


def get_files_mtime(directory):
    files_mtime = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            files_mtime[filename] = os.path.getmtime(filepath)
    return files_mtime


def check_changes(directory, last_mtime):
    current_mtime = get_files_mtime(directory)
    added_files = set(current_mtime.keys()) - set(last_mtime.keys())
    removed_files = set(last_mtime.keys()) - set(current_mtime.keys())
    modified_files = {filename for filename in current_mtime if
                      filename in last_mtime and current_mtime[filename] != last_mtime[filename]}

    return added_files, removed_files, modified_files, current_mtime


if __name__ == "__main__":
    directory = "logs"

    # 获取初始文件修改时间
    last_mtime = get_files_mtime(directory)
    print(last_mtime)

    # 等待一段时间后检查文件变化
    time.sleep(10)  # 等待10秒钟，实际使用中根据需要调整

    added, removed, modified, new_mtime = check_changes(directory, {})

    print("新增文件:", added)
    print("删除文件:", removed)
    print("修改文件:", modified)
    print(new_mtime)
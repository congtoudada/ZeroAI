import os
from PIL import Image


def convert_jpg_to_png(file_path):
    try:
        # 打开图片文件
        with Image.open(file_path) as img:
            # 构造新的文件名，替换.jpg为.png
            new_file_path = file_path.rsplit('.', 1)[0] + '.png'
            # 将图片保存为png格式
            img.save(new_file_path, 'PNG')
            print(f"Converted: {file_path} -> {new_file_path}")
            os.remove(file_path)
    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")


def traverse_and_convert(directory):
    # 遍历文件夹中的所有文件
    for root, _, files in os.walk(directory):
        for file in files:
            # 检查文件是否为jpg文件
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                convert_jpg_to_png(file_path)
                # os.remove(file_path)


if __name__ == "__main__":
    directory = rf"H:\AI\dataset\VAD\Featurize\shanghaitech\test"
    traverse_and_convert(directory)

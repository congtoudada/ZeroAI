import os
from PIL import Image


def convert_jpg_to_png(file_path):
    try:
        # 打开图片文件
        with Image.open(file_path) as img:
            # 构造新的文件名，替换.jpg为.png
            # new_file_path = file_path.rsplit('.', 1)[0] + '.png'
            # 获取文件名（不带路径和扩展名）
            base_name = os.path.basename(file_path)
            file_name_without_extension = os.path.splitext(base_name)[0]

            # 去除文件名中的前导零
            new_file_name = str(int(file_name_without_extension))  # 转换为整数后再转换回字符串，会去掉前导零

            # 构造新的文件路径，替换为.png扩展名
            new_file_path = os.path.join(os.path.dirname(file_path), f"{new_file_name}.png")
            # 将图片保存为png格式
            img.save(new_file_path, 'PNG')
            print(f"Converted: {file_path} -> {new_file_path}")

    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")


def traverse_and_convert(directory):
    # 遍历文件夹中的所有文件
    for root, _, files in os.walk(directory):
        for file in files:
            # 检查文件是否为jpg文件
            file_path = os.path.join(root, file)
            if file.lower().endswith('.tif'):
                convert_jpg_to_png(file_path)
                os.remove(file_path)
            else:
                if not file.lower().endswith('.png'):
                    os.remove(file_path)


if __name__ == "__main__":
    directory = rf"H:\AI\dataset\VAD\Featurize_png\ped2\test\frames"
    traverse_and_convert(directory)
    directory = rf"H:\AI\dataset\VAD\Featurize_png\ped2\train\frames"
    traverse_and_convert(directory)

from flask import Flask, request, jsonify
from reid import Reid
import os
import shutil

from collections import Counter
import json
import requests
app = Flask(__name__)

def my_algorithm(query_directory_or_id, gallery_directory):
    # 创建 Reid 类的实例
    reid_instance = Reid()
    

    if not isinstance(query_directory_or_id, str):
        return {"error": "Invalid input format, 'data1' should be a directory path."}
    if not os.path.isdir(query_directory_or_id):
        return {"error": f"Directory does not exist: {query_directory_or_id}"}
    if not isinstance(gallery_directory, str):
        return {"error": "Invalid input format, 'data2' should be a directory path."}
    if not os.path.isdir(gallery_directory):
        return {"error": f"Directory does not exist: {gallery_directory}"}

    # 调用 run 方法并传递目录路径
    q_topn_dir = reid_instance.run(query_directory_or_id, gallery_directory)

    # 返回结果
    return q_topn_dir

# 假设这是你的固定目录值
task3_tmp_queryPath = "res/images/reid_tmp_data/idBase_findLocation_query" 

@app.route('/process', methods=['POST'])
def process_request():
    print("调试代码 36")
    # 获取JSON数据
    # data = request.get_json()
    data = request.form
    
    # 'gallery_directory' 是可选的
    if 'gallery_directory' in data:
        gallery_directory = data['gallery_directory']
        
    #  'query_directory_or_id' 是必须的
    if 'query_directory_or_id' not in data:
        return jsonify({"error": "缺少 'query_directory_or_id' 参数。"}), 400
    query_directory_or_id = data['query_directory_or_id']


    # 需求3
        # 如果query_directory_or_id是整数，
        # 则认为是'int+地址'的情况，
        # 将整数赋值给id_int，并使用固定目录

    gallery_directory = "output/business/phone/timing"
    target_id = query_directory_or_id
    print("调试58")
    copy_images_with_id(target_id, task3_tmp_queryPath)

    q_topn_dir = my_algorithm(task3_tmp_queryPath, gallery_directory)
    result = result_task3(q_topn_dir)
    
    #json_result = json.dumps(result, indent=4)
    # 假设你想要写入到 'output.json' 文件
    # with open('task3.json', 'w') as json_file:
    #     json.dump(result, json_file, indent=4)
    print("调试68 返回最终结果")
    return result
    

@app.route('/process2', methods=['POST'])
def process_request2():
    print("调试代码 74")
    # 获取JSON数据
    data = request.get_json()
    # data = request.form
    
    print("调试代码 79")
    # 'gallery_directory' 是可选的
    if 'gallery_directory' in data:
        gallery_directory = data['gallery_directory']
        
    #  'query_directory_or_id' 是必须的
    if 'query_directory_or_id' not in data:
        return jsonify({"error": "缺少 'query_directory_or_id' 参数。"}), 400
    
    # 获取 'query_directory_or_id' 的值
    query_directory_or_id = data['query_directory_or_id']
    
    
    #需求4
    gallery_directory = "res/images/reid_tmp_data/id_gt"

    # # 确保query_directory_or_id是一个字符串路径
    # if not isinstance(query_directory_or_id):
    #     return jsonify({"error": "'query_directory_or_id' 应该是一个字符串表示的目录路径。"}), 400
    # if not os.path.isdir(query_directory_or_id):
    #     return jsonify({"error": f"'{query_directory_or_id}' 不是一个有效的目录路径。"}), 400

    print("调试101", query_directory_or_id,gallery_directory )
    q_topn_dir = my_algorithm(query_directory_or_id, gallery_directory)
    print("调试代码, httpdemo line87", q_topn_dir)
    result = result_task4(q_topn_dir)
    
    
    print("调试代码 104")

    # 发送给后端
    url = 'http://localhost:8080/algorithm/warn'
    headers = {'Content-Type': 'application/json'}

    print("开始Post请求")
    for warning_index, warning in result.items():
        response = requests.post(url, json=warning, headers=headers)
        if response.status_code == 200:
            print(f"Warning {warning_index} 发送成功")
        else:
            print(f"Warning {warning_index} 发送失败，状态码：{response.status_code}")

    
    # json_result = json.dumps(result, indent=4)
    # 假设你想要写入到 'output.json' 文件
    # with open('task4.json', 'w') as json_file:
    #     json.dump(result, json_file, indent=4)
    return result   

#############################################################################################################
def copy_images_with_id(image_id, target_directory):
    source_directory = "res/images/reid_tmp_data/id_gt" #更改
    # 确保目标目录存在
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    else:
        # 清空目标目录中的所有文件
        for filename in os.listdir(target_directory):
            file_path = os.path.join(target_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'无法删除 {file_path}. 原因: {e}')
    
    # 使用image_id作为四位数字符串
    image_id_str = f"{int(image_id):04}"
    
    # 遍历源目录中的所有文件
    for filename in os.listdir(source_directory):
        # 提取文件名中的 image_id 部分并格式化为四位数
        file_id = filename.split('_')[0]
        file_id_str = f"{int(file_id):04}"
        
        # 检查格式化后的文件 image_id 是否与传入的 image_id 相匹配
        if file_id_str == image_id_str:
            source_file = os.path.join(source_directory, filename)
            target_file = os.path.join(target_directory, filename)
            # 复制文件
            shutil.copy2(source_file, target_file)
            #print(f"复制图片：{source_file} 到 {target_file}")


def result_task3(q_topn_dir, k=4):
    # 功能1: 剔除掉所有的query
    filtered_data = [item for sublist in q_topn_dir for item in sublist[1:]]
    
    # 功能2: 计算重复次数并按照次数排序，选取前k个
    counts = Counter(filtered_data)  # 计算每个元素的出现次数
    most_common_data = counts.most_common(k)  # 获取前k个最常见的元素及其计数
    
    # 功能3: 将剩余内容整理成所需结构
    result = []
    for index, (image_path, _) in enumerate(most_common_data):
        # 从文件路径中提取文件名
        filename = os.path.basename(image_path)
        # 分割文件名以获取 person_id 和 camera_id
        parts = filename.split('_')
        camera_id = parts[1]
        time = parts[2]
        # 添加到结果列表中
        result.append({
            'shotImg': image_path,
            'camId': camera_id,
            'recordTime':time  
        })
    
    return result

def result_task4(q_topn_dir):
    warnings_dict = {}
    
    for warning_index, group in enumerate(q_topn_dir):
        # 功能1: 提取 warning_image 的完整路径
        warning_image_path = group[0]
        
        # 从路径中提取图片名
        _, image_name = os.path.split(warning_image_path)
        # 从图片名中提取 camera_id 和 time
        parts = image_name.split('_')
        camera_id = parts[1]
        time = parts[2]
        
        # 功能2: 统计除了第一个元素之外的 person_id 出现次数最多的
        person_ids = [os.path.basename(path).split('_')[0] for path in group[1:]]
        most_common_person_id, _ = Counter(person_ids).most_common(1)[0]  # 取出现次数最多的 person_id
        
        # 构建 warning 字典
        warning = {
            'shotImg': warning_image_path,
            'recordTime': time,
            'camId': camera_id,
            'personId': most_common_person_id,
            "warnType": 1
        }
        
        # 将 warning 字典添加到 warnings_dict 字典中，以 warning_index 作为键
        warnings_dict[warning_index] = warning

    return warnings_dict
#################################################################################################################################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    


from flask import Flask, request, jsonify
from reid import Reid
import os

app = Flask(__name__)

def my_algorithm(demo_data_directory1, demo_data_directory2):
    # 创建 Reid 类的实例
    reid_instance = Reid()
    

    if not isinstance(demo_data_directory1, str):
        return {"error": "Invalid input format, 'demo_data' should be a directory path."}
    if not os.path.isdir(demo_data_directory1):
        return {"error": f"Directory does not exist: {demo_data_directory1}"}
    if not isinstance(demo_data_directory2, str):
        return {"error": "Invalid input format, 'demo_data' should be a directory path."}
    if not os.path.isdir(demo_data_directory2):
        return {"error": f"Directory does not exist: {demo_data_directory2}"}
    
    

    # 调用 run 方法并传递目录路径
    result = reid_instance.run(demo_data_directory1, demo_data_directory2)

    # 返回结果
    return result

@app.route('/process', methods=['POST'])
def process_request():
    # 获取JSON数据
    data = request.get_json()
    
    # 检查 'demo_data1' 键是否存在，并且是一个字符串
    demo_data_directory1 = data.get('demo_data1')
    if demo_data_directory1 is None:
        return jsonify({"error": "Missing 'demo_data1' key."}), 400
    if not isinstance(demo_data_directory1, str):
        return jsonify({"error": "'demo_data1' should be a string representing a directory path."}), 400
    if not os.path.isdir(demo_data_directory1):
        return jsonify({"error": f"'{demo_data_directory1}' is not a valid directory path."}), 400
    # 检查 'demo_data2' 键是否存在，并且是一个字符串
    demo_data_directory2 = data.get('demo_data2')
    if demo_data_directory2 is None:
        return jsonify({"error": "Missing 'demo_data2' key."}), 400
    if not isinstance(demo_data_directory2, str):
        return jsonify({"error": "'demo_data2' should be a string representing a directory path."}), 400
    if not os.path.isdir(demo_data_directory2):
        return jsonify({"error": f"'{demo_data_directory2}' is not a valid directory path."}), 400
    
    # 运行算法
    result = my_algorithm(demo_data_directory1, demo_data_directory2)
    
    # 把结果转换为JSON，返回给客户端
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

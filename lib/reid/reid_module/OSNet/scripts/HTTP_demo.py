from flask import Flask, request, jsonify
from reid import Reid
import os

app = Flask(__name__)

def my_algorithm(demo_data_directory):
    # 创建 Reid 类的实例
    reid_instance = Reid()
    
    # 确保 demo_data_directory 是一个字符串
    if not isinstance(demo_data_directory, str):
        return {"error": "Invalid input format, 'demo_data' should be a directory path."}

    # 检查路径是否存在并且是一个目录
    if not os.path.isdir(demo_data_directory):
        return {"error": f"Directory does not exist: {demo_data_directory}"}

    # 调用 run 方法并传递目录路径
    result = reid_instance.run(demo_data_directory)

    # 返回结果
    return result

@app.route('/process', methods=['POST'])
def process_request():
    # 获取JSON数据
    #示例：curl -X POST -H "Content-Type: application/json" -d "{\"demo_data\": \"C:/Users/zuyi/Downloads/demo_data\"}" http://127.0.0.1:5000/process
    data = request.get_json()
    
    # 检查 'demo_data' 键是否存在，并且是一个字符串
    demo_data_directory = data.get('demo_data')
    if demo_data_directory is None:
        return jsonify({"error": "Missing 'demo_data' key."}), 400
    if not isinstance(demo_data_directory, str):
        return jsonify({"error": "'demo_data' should be a string representing a directory path."}), 400
    if not os.path.isdir(demo_data_directory):
        return jsonify({"error": f"'{demo_data_directory}' is not a valid directory path."}), 400
    
    # 运行算法
    #print("http_demo,line41,调试待删除")
    result = my_algorithm(demo_data_directory)
    #3print(result,"http_demo,line43,调试待删除")
    # 把结果转换为JSON，返回给客户端
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

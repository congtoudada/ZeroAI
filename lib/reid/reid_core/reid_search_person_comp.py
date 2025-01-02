from flask import Flask  # 导入模块

app = Flask(__name__)  # 注意这里前后都是双在下划线


@app.route('/')  # 设置网页的目录也就是 path
def hello_world():
    return "Hello World!"  # 返回Hello World!,也就是在网页上显示Hello World!


if __name__ == "__main__":
    app.run()

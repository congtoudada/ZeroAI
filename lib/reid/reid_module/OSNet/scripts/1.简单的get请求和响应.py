import requests

#url地址
baidu_url= "https://www.baidu.com/"

# 请求叫request
# 响应叫response
# get方法简单的就可以发送请求了
response = requests.get(baidu_url)
print(response)
# 响应的状态码
print(response.status_code)
# text属性获取报文实体的字符串形式
print(response.text)
# encoding属性查看响应内容的编码
print(response.encoding)
# content来获得bytes类型的报文实体
print(response.content)
# 将content转换成utf-8的字符串
print(response.content.decode('utf-8'))

# 第二种， 我们用encoding属性来控制响应的编码方式
response.encoding = 'utf-8'
print(response.text)

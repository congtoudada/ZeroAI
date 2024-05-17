import asyncio
import threading
import time
import httpx


class WebKit(object):
    Initialized = False
    Prefix_url = "http://localhost:9012/algorithm"

    @staticmethod
    def initialize():
        if not WebKit.Initialized:
            WebKit.Initialized = True
            # 在新的线程中运行事件循环
            thread = threading.Thread(target=asyncio.get_event_loop().run_forever, daemon=True)
            thread.start()

    @staticmethod
    async def _async_get(url):
        async with httpx.AsyncClient() as client:
            await client.get(url)
            # response = await client.get(url)
            # print(response.text)
            # return response.text

    @staticmethod
    def get(url):
        if not WebKit.Initialized:
            WebKit.initialize()
        asyncio.get_event_loop().create_task(WebKit._async_get(url))

    @staticmethod
    async def _async_post(url, data: dict):
        async with httpx.AsyncClient() as client:
            headers = {'Content-Type': 'application/json'}
            await client.post(url, json=data, headers=headers)
            # response = await client.post(url, json=data, headers=headers)
            # print(response.status_code)
            # print(response.text)

    @staticmethod
    def post(url, data: dict):
        if not WebKit.Initialized:
            WebKit.initialize()
        asyncio.get_event_loop().create_task(WebKit._async_post(url, data))


if __name__ == "__main__":
    url = "http://localhost:9012/unity/hotfix"
    WebKit.initialize()
    # print("开始Get请求")
    # WebKit.get(url)
    url = "http://localhost:9012/unity/hotfix2"
    print("开始Post请求")
    WebKit.post(url, {"key": 10, "pageType": 1})
    # WebKit.post(url, {"DTO": 10})
    time.sleep(3)

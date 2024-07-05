import asyncio
import threading
import time
import httpx
from loguru import logger


class WebKit(object):
    Initialized = False
    # Prefix_url = "http://210.30.97.235:8080/algorithm"
    # Prefix_url = "http://localhost:9012/algorithm"
    Prefix_url = "http://localhost:8080/algorithm"
    get_queue = []
    post_queue = []

    @staticmethod
    def initialize():
        if not WebKit.Initialized:
            WebKit.Initialized = True
            # 在新的线程中运行事件循环
            thread = threading.Thread(target=asyncio.get_event_loop().run_forever, daemon=True)
            thread.start()
            asyncio.get_event_loop().create_task(WebKit._async_request())
            # asyncio.get_event_loop().create_task(WebKit._async_get())

    # @staticmethod
    # async def _async_get():
    #     async with httpx.AsyncClient() as client:
    #         while True:
    #             while len(WebKit.get_queue) > 0:
    #                 url = WebKit.get_queue.pop()
    #                 # await client.get(url)
    #                 response = await client.get(url)
    #                 print(response.status_code)
    #             time.sleep(0.5)
    #
    @staticmethod
    def get(url):
        if not WebKit.Initialized:
            WebKit.initialize()
        WebKit.get_queue.append(url)

    @staticmethod
    async def _async_request():
        async with httpx.AsyncClient() as client:
            while True:
                while len(WebKit.post_queue) > 0:
                    (url, data) = WebKit.post_queue.pop()
                    headers = {'Content-Type': 'application/json'}
                    # await client.post(url, json=data, headers=headers)
                    response = await client.post(url, json=data, headers=headers)
                    logger.info(f"web response: {response.status_code}")
                    # print(response.text)
                while len(WebKit.get_queue) > 0:
                    url = WebKit.get_queue.pop()
                    # await client.get(url)
                    response = await client.get(url)
                    logger.info(f"web response: {response.status_code}")
                time.sleep(0.5)


    @staticmethod
    def post(url, data: dict):
        if not WebKit.Initialized:
            WebKit.initialize()
        WebKit.post_queue.append((url, data))



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

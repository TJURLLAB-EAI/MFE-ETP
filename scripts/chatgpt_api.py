import sys
import time
import traceback

import torch
from openai import OpenAI
import base64
import requests
import os
import json
import numpy as np

def read_config(path):
    with open(path) as json_file:
        config = json.load(json_file)
    return config


cfg = read_config("openai_cfg.json")["apis"][0]

API_KEY = cfg["api_key"]

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["http_proxy"] = cfg["http_proxy"]
os.environ["https_proxy"] = cfg["https_proxy"]


Proxy = cfg["proxy"]


class Chat:

    def __init__(self, model="gpt-4-vision-preview"):
        self.model = model
        self.api_key = API_KEY

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def answer(self, raw_img_list, context, max_tokens=300):
        if not isinstance(context, str):
            context = context[0]
        assert isinstance(context, str)

        base64_image_list = []
        raw_img_list = np.array(raw_img_list).flatten()
        for i in range(len(raw_img_list)):
            base64_image_list.append(self.encode_image(raw_img_list[i]))
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": str(self.model),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": context
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        for i in range(len(base64_image_list)):
            payload["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_list[i]}"
                }
            })

        for i in range(200):
            try:
                if i >= 1:
                    time.sleep(0.5)
                    print(f"Retrying times: {i}")
                    
                response = requests.post("https://"+Proxy+"/v1/chat/completions", headers=headers, json=payload, timeout=500)

                if response.status_code == requests.codes.ok:
                    break
                else:
                    print(f"response.status_code:{response.status_code}")

            except Exception as e:
                print(e)  # division by zero
                print(sys.exc_info())
                print('\n', '>>>' * 20)
                print(traceback.print_exc())
                print('\n', '>>>' * 20)
                print(traceback.format_exc())

        response_list = []
        response_list.append(response.json()['choices'][0]['message']['content'])
        return response_list

import json
import os
import base64
import re
import sys
import time
import traceback

import requests


def read_config(path):
    with open(path) as json_file:
        config = json.load(json_file)
    return config


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_index(dict_):
    for i in range(len(dict_["data"])):
        dict_["data"][i]["sample_id"] = i
    return dict_


def process_num_sample(dict_):
    dict_["metadata"]["num_sample"] = len(dict_["data"])
    return dict_


def save_to_json(dict1, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(dict1, f, indent=4, ensure_ascii=False)


def auto_save_file(path):
    if not os.path.exists(path):
        return path

    directory, file_name = os.path.split(path)
    while os.path.isfile(path):
        pattern = '(\d+)\)\.'
        if re.search(pattern, file_name) is None:
            file_name = file_name.replace('.txt', '(0).txt')
        else:
            current_number = int(re.findall(pattern, file_name)[-1])
            new_number = current_number + 1
            file_name = file_name.replace(f'({current_number}).', f'({new_number}).')
        path = os.path.join(directory + os.sep + file_name)
    print(f"Warning: prompt file already exist\nNew file name:{path}")
    return path


def check_image_existence(dict1, image_path):
    cnt = 0
    for i in range(len(dict1["data"])):
        for image_name in dict1["data"][i]["task_instance"]["images_path"]:
            # print(f"opening {image_name}:")
            absolute_path = image_path + image_name
            if not os.path.exists(absolute_path):
                print(f"image {image_name} do not exists")
                cnt += 1
    if cnt == 0:
        print("all exist")
        return True
    return False

cfg = read_config("openai_cfg.json")["apis"][0]

api_key = cfg["api_key"]

os.environ["OPENAI_API_KEY"] = api_key
os.environ["http_proxy"] = cfg["http_proxy"]
os.environ["https_proxy"] = cfg["https_proxy"]


Proxy = cfg["proxy"]

### task_planning
dim = "task_planning4"
image_root = "./data/task_planning/images/"

output_dir = "./benchmark-evaluation/task_planning/trial1"
prompt_root = os.path.join(output_dir, "prompt")
result_root = os.path.join(output_dir, "result")

data_path = "./data/task_planning/data.json"

# Pre process
datas = read_config(data_path)

process_index(datas)
process_num_sample(datas)
save_to_json(datas, data_path)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(prompt_root):
    os.makedirs(prompt_root)
if not os.path.exists(result_root):
    os.makedirs(result_root)

######################################################################################
                             #      Begin      #
######################################################################################

OFFSET = 0

results = []

question_tp = f"""
There is a dual armed humanoid robot (agent) that can perform a variety of household tasks just like a human. The robot is currently ready to perform a household task called "task_name". 

The language description of the task goal is:
task_description

I'll give you an image. Note that the image represents robot's current perspective. Please complete the following two steps according to the given image and the above task goal. 

Step 1:  The given image represents robot's current perspective.  To ensure that the robot can accomplish the above task goal, output the correct action plan for the robot to execute using a set of predefined action functions. At the same time, output the specific preconditions and postconditions of each action function in the action plan. If the arguments of action functions are objects, specify the location of the objects. 

To output the correct action plan to enable the robot to complete the task, please accurately identify the task-related objects involved in the current perspective and the objects not in the current perspective but related to the task.

To output the correct action plan, please consider the initial position of the robot and the spatial position relationship between the robot and various objects in robot's current perspective. For example, when the robot is in the initial position, some objects in robot's current perspective may or may not be within reachable distance for the robot. However, when the robot is in the initial position, objects outside robot's current perspective are not be within reachable distance for the robot.  The robot position may change as the robot gradually performs the action according to the action plan. When planning each action, consider the position of the robot and its position relationship to the relevant objects.

It is worth noting that the following predefined action functions can be adopted in the action plan if and only if their preconditions are met. For example, if the output action plan has a step, navigate_to(arg1), it means that the "arg1" is not within reachable distance for the robot before executing this step.
Predefined action functions:
1. navigate_to(arg1): "Navigate to the arg1, which can be a object or a room.  Preconditions: arg1 is not within reachable distance for the robot. Postconditions: arg1 is within reachable distance for the robot. "
2. grasp(arg1): "Grasp arg1. Preconditions: arg1 is within reachable distance and no object is currently held. Postconditions: arg1 is being held."
3. place_onTop(arg1, arg2): "Place arg1 on top of arg2. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions: arg1 is put on top of arg2."
4. place_under(arg1, arg2):"Place arg1 under arg2. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions:arg1 is put under arg2."
5. place_onLeft(arg1, arg2): "Place arg1 on left of arg2. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions: arg1 is put on left of arg2."
6. place_onRight(arg1, arg2): "Place arg1 on right of arg2. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions: arg1 is put on right of arg2."
7. place_inside(arg1, arg2): "Place arg1 inside of arg2. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions:arg1 is put inside of arg2."
8. open(arg1): "Open arg1 . Preconditions: arg1 is closed, and arg1 is reachable . Postconditions: arg1 is open."
9. close(arg1): "Close arg1 . Preconditions: arg1 is open, and arg1 is reachable . Postconditions: arg1 is closed."
10. slice(arg1): "Slice arg1. Preconditions: arg1 is not sliced, and arg1 is reachable . Postconditions: arg1 is sliced"
11. wipe(arg1, arg2):"Wipe across the surface of arg2 with arg1. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions: arg1 continues to be held, arg2 holds state unchanged."
12. wait(arg1): "Wait for arg1 seconds. Preconditions: None . Postconditions: arg1 second(s) has(have) passed."
13. toggle(arg1): "Press the button of arg1 to turn it on or off，Preconditions: arg1 is open/closed, and arg1 is reachable . Postconditions: arg1 is closed/open."
14. water(arg1, arg2):"Water arg2 with arg1. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions: arg1 continues to be held, arg2 is watered."
Note: To obtain the correct action plan, multiple action function calls may be called, and the arguments are specified as the names of related objects (except for the 'wait' function where the argument is a number representing the number of seconds). Creating or calling other functions is prohibited.

Below I provide you with an output example.

1. navigate_to (jar.1) 
Preconditions: jar.1 is on the kitchen counter and it is not within reachable distance for the robot. 
Postconditions: jar.1 is within reachable distance for the robot. 

2. grasp (jar.1)
Preconditions: jar.1 is on the kitchen counter. jar.1 is within reachable distance and no object is currently held. 
Postconditions: jar.1 is being held.

Step 2: Assume that the robot has already performed the following steps in the past. To ensure that the robot can accomplish the above task goal, output the subsequent correct action plan for the robot to execute using the above a set of predefined action functions. At the same time, output the specific preconditions and postconditions of each action function in the action plan. If the arguments of action functions are objects, specify the location of the objects. 

The steps performed in the past are:
past_steps
"""

question = f"""
There is a dual armed humanoid robot (agent) that can perform a variety of household tasks just like a human, but the robot can only grab one item at a time. The robot is currently ready to perform a household task called "task_name". 

The language description of the task goal is:
task_description

I'll give you an image. Note that the image represents robot's current perspective. Please complete the following one steps according to the given image and the above task goal. 

Step 1:  The given image represents robot's current perspective.  To ensure that the robot can accomplish the above task goal, output the correct action plan for the robot to execute using a set of predefined action functions. At the same time, output the specific preconditions and postconditions of each action function in the action plan. If the arguments of action functions are objects, specify the location of the objects. 

To output the correct action plan to enable the robot to complete the task, please accurately identify the task-related objects involved in the current perspective and the objects not in the current perspective but related to the task.

To output the correct action plan, please consider the initial position of the robot and the spatial position relationship between the robot and various objects in robot's current perspective. For example, when the robot is in the initial position, some objects in robot's current perspective may or may not be within reachable distance for the robot. However, when the robot is in the initial position, objects outside robot's current perspective are not be within reachable distance for the robot.  The robot position may change as the robot gradually performs the action according to the action plan. (For example, when the robot have completed the action, place_inside(pie, fridge), the robot position moves from its previous position to next to the fridge. If robot want to return to the previous position, robot need to perform the navigation action) When planning each action, consider the position of the robot and its position relationship to the relevant objects.

It is worth noting that the following predefined action functions can be adopted in the action plan if and only if their preconditions are met. For example, if the output action plan has a step, navigate_to(arg1), it means that the "arg1" is not within reachable distance for the robot before executing this step.
Predefined action functions:
1. navigate_to(arg1): "Navigate to the arg1, which can be a object.  Preconditions: arg1 is not within reachable distance for the robot. Postconditions: arg1 is within reachable distance for the robot. "
2. grasp(arg1): "Grasp arg1. Preconditions: arg1 is within reachable distance and no object is currently held. Postconditions: arg1 is being held, and the robot's position is unchanged compared to the robot's position before the action is performed."
3. place_onTop(arg1, arg2): "Place arg1 on top of arg2. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions: arg1 is put on top of arg2. The robot's position is unchanged compared to the robot's position before the action is performed and arg2 is reachable."
4. place_under(arg1, arg2):"Place arg1 under arg2. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions:arg1 is put under arg2. The robot's position is unchanged compared to the robot's position before the action is performed and arg2 is reachable."
5. place_onLeft(arg1, arg2): "Place arg1 on left of arg2. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions: arg1 is put on left of arg2. The robot's position is unchanged compared to the robot's position before the action is performed and arg2 is reachable."
6. place_onRight(arg1, arg2): "Place arg1 on right of arg2. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions: arg1 is put on right of arg2. The robot's position is unchanged compared to the robot's position before the action is performed and arg2 is reachable."
7. place_inside(arg1, arg2): "Place arg1 inside of arg2. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions:arg1 is put inside of arg2. The robot's position is unchanged compared to the robot's position before the action is performed and arg2 is reachable."
8. open(arg1): "Open arg1 . Preconditions: arg1 is closed, and arg1 is reachable . Postconditions: arg1 is open."
9. close(arg1): "Close arg1 . Preconditions: arg1 is open, and arg1 is reachable . Postconditions: arg1 is closed."
10. slice(arg1): "Slice arg1. Preconditions: arg1 is not sliced, and arg1 is reachable . Postconditions: arg1 is sliced"
11. wipe(arg1, arg2):"Wipe across the surface of arg2 with arg1. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions: arg1 continues to be held, arg2 holds state unchanged."
12. wait(arg1): "Wait for arg1 seconds. Preconditions: None . Postconditions: arg1 second(s) has(have) passed."
13. toggle(arg1): "Press the button of arg1 to turn it on or off，Preconditions: arg1 is open/closed, and arg1 is reachable . Postconditions: arg1 is closed/open."
14. water(arg1, arg2):"Water arg2 with arg1. Preconditions: arg1 is currently being held, and arg2 is reachable . Postconditions: arg1 continues to be held, arg2 is watered."
Note: To obtain the correct action plan, multiple action function calls may be called, and the arguments are specified as the names of related objects (except for the 'wait' function where the argument is a number representing the number of seconds). Creating or calling other functions is prohibited.

Below I provide you with an output example.

1. navigate_to (jar.1) 
Preconditions: jar.1 is on the kitchen counter and it is not within reachable distance for the robot. 
Postconditions: jar.1 is within reachable distance for the robot. 

2. grasp (jar.1)
Preconditions: jar.1 is on the kitchen counter. jar.1 is within reachable distance and no object is currently held. 
Postconditions: jar.1 is being held.
"""

for i in range(len(datas["data"])-OFFSET):

    task_name = datas["data"][i+OFFSET]["task_instance"]["task_name"]
    task_description = datas["data"][i+OFFSET]["task_instance"]["task_description"]

    print(f"task_name:{task_name}")
    print(f"task_description:{task_description}")

    # create output txt file name
    prompt_path = auto_save_file(os.path.join(prompt_root, str(datas["data"][i+OFFSET]["sample_id"]) + "." + task_name + ".txt"))
    print(prompt_path)
    result_path = auto_save_file(os.path.join(result_root, str(datas["data"][i+OFFSET]["sample_id"]) + "." + task_name + ".txt"))
    print(result_path)

    if dim == "time perception":
        question = question_tp.replace('task_name', task_name)
    else:
        question = question.replace('task_name', task_name)
    question = question.replace('task_description', task_description)

    with open(prompt_path, 'w', encoding='utf8') as f:
        print(question, file=f)

    # images
    base64_image_list = []
    for p in datas["data"][i+OFFSET]["task_instance"]["images_path"]:
        img_path = os.path.join(image_root, p)
        base64_image_list.append(encode_image(img_path))

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        "max_tokens": 3000
    }

    # add images
    for img in base64_image_list:
        payload["messages"][0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img}"
            }
        })

    # asking GPT
    for j in range(10):
        try:
            if j >= 1:
                time.sleep(0.5)
                print(f"Retrying times: {j}")

            print("posting...")
            response = requests.post("https://" + Proxy + "/v1/chat/completions", headers=headers, json=payload,
                                     timeout=500)
            print(response)

            if response.status_code == requests.codes.ok:
                break
            else:
                print(f"wrong response.status_code:{response.status_code}")
                if response.status_code == 429:
                    print("*************************Run out of money*************************")
                    print(response.json())

        except Exception as e:
            print(e)  # division by zero
            print(sys.exc_info())
            print('\n', '>>>' * 20)
            print(traceback.print_exc())
            print('\n', '>>>' * 20)
            print(traceback.format_exc())

    results.append({'sample_id': datas["data"][i+OFFSET]["sample_id"], 'prompt': question, 'pred_response': response.json()['choices'][0]['message']['content']})

    with open(result_path, 'w', encoding='utf8') as f:
        print(response.json()['choices'][0]['message']['content'], file=f)



with open(os.path.join(output_dir, 'pred.json'), 'w', encoding='utf8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

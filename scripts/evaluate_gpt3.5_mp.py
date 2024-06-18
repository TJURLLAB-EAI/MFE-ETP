import re
import argparse
import os
import json
import sys
import time
import traceback

import threading
import numpy as np

import requests


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

class Eval:
    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                    re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    # revise for gpt evaluation
    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = answer.strip().lower()
        return answer

    # gpt-3.5-turbo-1106
    def get_gpt_scores(self, question, pred_ans, gt_ans, dataset, acc, eval_list, sample_id, max_tokens=300, model="gpt-3.5-turbo-1106"):
        print(f"#############################\nsample_id: {sample_id}")
        if pred_ans == '':
            s = 0
            a = "pred_ans is none"
        else:
            context_dict = {
                'object_type': "Now, you will be presented with a correct response, and a student's answer to a question about the type of objects. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. (The student's answer is very close to the correct answer, for example, they are all objects of the same type, with similar appearance and purposes) If the answer is incorrect, give it a '0'. If the answer contains correct and incorrect objects, give it a '0.5'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0, or 0.5)', which is your final judgement. Output in JSON format. For instance: {\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0, or 0.5)\"}. The correct response and student's answer is provided below. ",
                'property': "Now, you will be presented with a correct response, and a student's answer to a question about the properties of objects. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer contains correct and incorrect properties, give it a '0.5'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0, or 0.5)', which is your final judgement. Output in JSON format. For instance: {\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0, or 0.5)\"}. The correct response and student's answer is provided below. ",
                'sequential': "Now, you will be presented with a correct response, and a student's answer to some question. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer contains correct and incorrect options, give it a '0'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)', which is your final judgement. Output in JSON format. For instance: {\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0)\"}. The correct response and student's answer is provided below.",
                'operation': "Now, you will be presented with a correct response, and a student's answer to some question. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer contains correct and incorrect options, give it a '0'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)', which is your final judgement. Output in JSON format. For instance: {\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0)\"}. The correct response and student's answer is provided below.",
                'spatial_relationship': "Now, you will be presented with a correct response, and a student's answer to a question about the spatial relationship between two objects. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer contains correct and incorrect options, give it a '0'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0, or 0.5)', which is your final judgement. Output in JSON format. For instance: {\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0, or 0.5)\"}. The correct response and student's answer is provided below. ",
                'task_planning': "Now, you will be presented with a correct response, and a student's answer to a question about task planning. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer contains correct and incorrect options, give it a '0'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Note that the provided correct answer is just one solution, and there may be multiple reasonabe task planning solutions for the same problem. The order of subtasks of other correct answer can differ from the provided correct answer, as long as the task can be completed reasonably. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)', which is your final judgement. Output in JSON format. For instance: {\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0)\"}. The correct response and student's answer is provided below.",
                'default': "Now, you will be presented with a correct response, and a student's answer to some question. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer contains correct and incorrect options, give it a '0.5'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0, or 0.5)', which is your final judgement. Output in JSON format. For instance: {\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0, or 0.5)\"}. The correct response and student's answer is provided below. "
            }

            if dataset in context_dict.keys():
                context = context_dict[dataset]
                print(f"using {dataset} specific instruction")
            else:
                context = context_dict['default']
                print(f"using DEFAULT instruction")
            qa_info = f"\nThe Student’s Answer is:\"{pred_ans}\"\nThe Correct Answer is:\"{gt_ans}\""
            context = context + qa_info

            print(f"get_gpt_scores context: {context}\n")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }

            payload = {
                "model": str(model),
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You, as an AI system, have been equipped with a vast amount of knowledge and an understanding of logical reasoning. Your task is to use these capabilities to assess academic responses."
                            }
                        ]
                    },
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

            for i in range(200):
                try:
                    if i >= 1:
                        time.sleep(0.5)
                        print(f"Retrying times: {i}")
                    response = requests.post("https://" + Proxy + "/v1/chat/completions", headers=headers, json=payload, timeout=500)

                    output = eval(response.json()['choices'][0]['message']['content'])
                    assessment = output["Assessment"]
                    score = float(output["Final Score"])

                    if response.status_code == requests.codes.ok:
                        break
                except Exception as e:
                    print(e)  # division by zero
                    print(sys.exc_info())
                    print('\n', '>>>' * 20)
                    print(traceback.print_exc())
                    print('\n', '>>>' * 20)
                    print(traceback.format_exc())
            s = score
            a = assessment


        acc['f'].append(s)
        eval_list.append({'id': str(sample_id), 'score': str(round(s, 3)), 'assessment': a})
        print(eval_list[-1])
        return score

    def evaluate_gpt(self, preds, dataset, datas):
        acc = {'f': [], }
        eval_list = []
        NUM_PROCESS = 10
        try:
            for i in range(0, len(preds), NUM_PROCESS):
                # mp:
                Threads = []
                for j in range(NUM_PROCESS):
                    # print(i+j)
                    if i + j >= len(preds):
                        break
                    sample_id = preds[i+j]['sample_id']
                    gt_ans = self.process(preds[i+j]["gt_response"])
                    pred_ans = self.process(preds[i+j]["pred_response"])
                    question = datas[i+j]["task_instance"]["context"]
                    assert gt_ans != ''
                    t = threading.Thread(target=self.get_gpt_scores, args=((question, pred_ans, gt_ans, dataset, acc, eval_list, sample_id)))
                    t.setDaemon(True)
                    Threads.append(t)
                for t in Threads:
                    t.start()
                for t in Threads:
                    t.join()

        finally:
            results = {'gpt-3.5': np.mean(acc['f'])}
            eval_list.sort(key=lambda x:int(x["id"]))
            print(f"\n\n\n***************\neval_list\n***************\n{json.dumps(eval_list)}")
            print(f"\n\n\n***************\nresults\n***************\n{results}")
        return results, eval_list


parser = argparse.ArgumentParser()
parser.add_argument('--project-dir', type=str, required=False, default="./data")
parser.add_argument("--result-dir", type=str, required=False, default="./benchmark-evaluation/")
args = parser.parse_args()


def evaluate_dataset(args, dataset, model):
    project_dir = args.project_dir
    result_dir = args.result_dir + model
    model_name = model
    core_annotation = json.load(open(os.path.join(project_dir, dataset, 'data.json'), 'r', encoding='utf8'))
    datas = core_annotation['data']

    E = Eval()
    output_dir = os.path.join(result_dir, dataset)
    if not os.path.exists(os.path.join(output_dir, 'pred.json')):
        print('%s--%s  No prediction file found' % (model_name, dataset))
        return
    preds = json.load(open(os.path.join(output_dir, 'pred.json'), 'r', encoding='utf8'))

    eval_result, eval_list = E.evaluate_gpt(preds, dataset, datas=datas)

    print(model_name, end=':  ')
    print(dataset, end=':  ')
    print(eval_result)
    with open(os.path.join(output_dir, 'eval_gpt3.5.json'), 'w') as f:
        json.dump(eval_result, f)

    with open(os.path.join(output_dir, 'eval_score_gpt3.5.json'), 'w') as f:
        json.dump(eval_list, f, indent=4)


def evaluate_model(args, model):
    dataset_list = ["goal", "sequence"]
    for dataset in dataset_list:
        start = time.time()
        evaluate_dataset(args, dataset, model)
        print(f"time:{time.time()-start}")

model_list = ["blip2flant5xl"]       # ["MiniCPM-Llama3-V-2_5", "instructblipflant5xl", "llava-v1.5-7b", "minigpt4-7b", "gpt-4-vision-preview", "blip2flant5xl"]
for model in model_list:
    print(f"Begin evaluating {model}")
    evaluate_model(args, model)
    print(f"Finish evaluating {model}")

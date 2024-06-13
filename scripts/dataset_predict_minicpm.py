import base64
import random
import json
import argparse
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from chat import OmniLMMChat, img2base64

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark prediction")
    parser.add_argument("--cfg-path", required=False, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument('--batch-image', type=int, required=False, default=30)
    parser.add_argument('--project-dir', type=str, required=False, default='./data')
    parser.add_argument("--result-dir", type=str, required=False, default="./benchmark-evaluation/MiniCPM-Llama3-V-2_5") #
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args


def setup_seeds(seed = 50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def collate_fn(batch):
    batch_data = {}
    batch_data['sample_id'] = [sample['sample_id'] for sample in batch]
    batch_data['context'] = [sample['context'] for sample in batch]
    batch_data['raw_img_list'] = [sample['raw_img_list'] for sample in batch]
    batch_data['response'] = [sample['response'] for sample in batch]

    return batch_data


# 按有几张图片 (n_img) 分割prompt
def split_data(data):
    data_dict = {}
    for d in data:
        n_img = len(d['task_instance']['images_path'])
        if n_img in data_dict:
            data_dict[n_img].append(d)
        else:
            data_dict[n_img] = [d]
    return data_dict


def load_to_dic(path):
    return json.load(open(path, 'r', encoding='utf-8'))


def get_ids(preds):
    id_list = []
    for pred in preds:
        id_list.append(pred["sample_id"])
    return id_list

def process_image(raw_list, img_root):
    img_path_list = np.array(raw_list).flatten()
    images_list = []
    flag = True
    for p in img_path_list:
        if flag:
            img = cv2.imread(os.path.join(img_root, p))
            images_list.append(img)
            width = img.shape[1]
            height = img.shape[0]
            img_size = (width, height)
            print("size:", img_size)
            flag = False
        else:
            img = cv2.imread(os.path.join(img_root, p))
            print(f"before:{img.shape}")
            img = cv2.resize(img, img_size, 0, 0, cv2.INTER_LINEAR)
            print(f"resize:{img.shape}")
            images_list.append(img)
    im_v = cv2.vconcat(images_list)

    retval, buffer = cv2.imencode('.jpg', im_v)
    if not retval:
        print("Could not encode image.")
        return None

    im_64 = base64.b64encode(buffer)

    return im_64

class IDataset(Dataset):
    def __init__(
            self, annoation, metadata, img_dir,
    ):
        self.img_dir = img_dir
        self.annotation = annoation
        #
        self.task_instructions = metadata['task_instruction']

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        task_instruction = self.task_instructions[ann['task_instruction_id']]
        context = task_instruction + ann['task_instance']['context']
        raw_img_list = []
        if 'choice_list' in ann['task_instance'].keys():
            choice_str = 'Choice list:[\'' + '\', \''.join(ann['task_instance']['choice_list']) + '\']. Your answer is:'
            context += choice_str
        for p in ann['task_instance']['images_path']:
            img_path = os.path.join(self.img_dir, p)
            raw_img_list.append(img_path)
        return {
            "sample_id": ann['sample_id'],
            "context": context,
            "raw_img_list": raw_img_list,
            "response": str(ann['response'])
        }


args = parse_args()


def predict(args, dataset):
    project_dir = args.project_dir
    dataset_name = dataset
    dataset_dir = os.path.join(project_dir, dataset_name)
    img_dir = os.path.join(dataset_dir, 'images')
    output_dir = os.path.join(args.result_dir, dataset_name)
    model_name = args.result_dir.split('/')[-1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    core_annotation = json.load(open(os.path.join(dataset_dir, 'data.json'), 'r'))  # encoding='utf-8' for open
    prediction_results = []
    data_dict = split_data(core_annotation['data'])

    chat_model = OmniLMMChat('openbmb/MiniCPM-Llama3-V-2_5')

    print('Initialization Finished')
    print('Predicting %s Using %s' % (dataset_name, model_name))
    try:
        for n_img, sub_data in data_dict.items():
            print('Proceeding %d-length images samples | Num:%d' % (n_img, len(sub_data)))
            E = IDataset(sub_data, core_annotation['metadata'], img_dir)
            data_loader = torch.utils.data.DataLoader(dataset=E, batch_size=1, shuffle=False, collate_fn=collate_fn)
            for i, samples in enumerate(tqdm(data_loader)):
                im_64 = process_image(samples['raw_img_list'], img_dir)
                context = samples['context']
                if not isinstance(context, str):
                    context = context[0]
                assert isinstance(context, str)
                msgs = [{"role": "user", "content": context}]
                inputs = {"image": im_64, "question": json.dumps(msgs)}
                answer = chat_model.chat(inputs)
                pred_responses = [answer]
                for sid, gt, p in zip(samples['sample_id'], samples['response'], pred_responses):
                    if torch.is_tensor(sid):
                        sid = sid.item()
                    prediction_results.append({'sample_id': sid, 'pred_response': p, 'gt_response': gt})
                    print(sid, p)

    except Exception as e:
        print("\n\nexcept\n\n")
        print(e)  # division by zero
        print(sys.exc_info())
        print('\n', '>>>' * 20)
        print(traceback.print_exc())
        print('\n', '>>>' * 20)
        print(traceback.format_exc())

    finally:
        print("\n\nfinally\n\n")
        print(f"**********************************************\nprediction_results:{prediction_results}\n**********************************************")
        prediction_results.sort(key=lambda x: x["sample_id"])
        with open(os.path.join(output_dir, 'pred.json'), 'w', encoding='utf8') as f:
            json.dump(prediction_results, f, indent=4, ensure_ascii=False)


dataset_list = ["goal", "sequence"]   #  ["object_type", "property", "time_perception", "spatial_perception", "operation", "goal", "sequential", "parallel"]                  #  "goal" ["object_type", "property", "time_perception", "spatial_perception", "relevant_object", "grasp_position"]
for dataset in dataset_list:
    predict(args, dataset)

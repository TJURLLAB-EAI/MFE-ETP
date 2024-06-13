import os
import torch
import pickle
# import pandas as pd
import numpy as np
import argparse
from torch.utils.data import Dataset
import json
import time
from tqdm import tqdm 
import sys
from PIL import Image
from torch.utils.data.dataloader import default_collate
import torch
from concurrent.futures import ThreadPoolExecutor
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.models import *
from minigpt4.processors import *

class EDataset(Dataset):
    def __init__(
        self, annoation, task_instructions, img_dir,
    ):
        self.img_dir = img_dir
        self.annotation = annoation
        self.task_instructions = task_instructions
    def __len__(self):
        return len(self.annotation)

    
    def __getitem__(self, index):
        ann = self.annotation[index]
        task_instruction = self.task_instructions[ann['task_instruction_id']]
        context = task_instruction + ann['task_instance']['context']
        raw_img_list = []
        for i in range(len(ann['task_instance']['images_path'])):
            rmv_txt = '{image#%d}'% (i+1)
            rmv_tbl = '{table#%d}'% (i+1)
            context = context.replace(rmv_txt, '<ImageHere>')
            context = context.replace(rmv_tbl, '<ImageHere>')
        for p in ann['task_instance']['images_path']:
            img_path = os.path.join(self.img_dir, p)
            raw_img = Image.open(img_path).convert('RGB')
            raw_img_list.append(raw_img)
        return {
            "sample_id": ann['sample_id'],
            "context": context,
            "raw_img_list": raw_img_list,
            "response": str(ann['response'])
        }        
def collate_fn(batch):
    batch_data={}
    batch_data['sample_id'] = [sample['sample_id'] for sample in batch]
    batch_data['context'] = [sample['context'] for sample in batch]
    batch_data['raw_img_list'] = [sample['raw_img_list'] for sample in batch]
    batch_data['response'] = [sample['response'] for sample in batch]
        
    return batch_data

def split_data(data):
    data_dict = {}
    for d in data:
        n_img = len(d['task_instance']['images_path'])
        if n_img in data_dict:
            data_dict[n_img].append(d)
        else:
            data_dict[n_img] = [d]
    return data_dict

def check_data(vis_root, data):
    def check_img_path(p):
        img_path = os.path.join(vis_root, p)
        if not os.path.isfile(img_path):
            print(f"{img_path} not exist")
            return False
        try:
            Image.open(img_path).convert("RGB")
        except Exception as e:
            print(img_path+' Failed! Exception:'+e)
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        all_images_exist = all(executor.map(check_img_path, [p for ann in data for p in ann['task_instance']['images_path']]))

    if all_images_exist:
        print('All images exist')
    else:
        exit(-1)

def check_prompt(prompt):
    if len(prompt) < 6:
        print('No prompt')
        exit(-1)

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=False, default=1)
parser.add_argument('--model', type=str, default="minigpt4-7b")
parser.add_argument('--output-root', type=str, default='./benchmark-evaluation')
parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

args = parser.parse_args()
batch_size = args.batch
model_name = args.model


cfg = Config('./minigpt4_eval.yaml')

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


def infer_dataset(dataset):

    dataset_dir = './data'
    task_dir = os.path.join(dataset_dir, dataset)
    img_dir  = os.path.join(task_dir,'images')

    output_dir = os.path.join('./benchmark-evaluation', model_name, dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_annotation = json.load(open(os.path.join(task_dir,'data.json'),'r'))
    data_dict = split_data(test_annotation['data'])

    check_data(img_dir,test_annotation['data'])
    check_prompt(test_annotation['metadata']['task_instruction'])
    for n_img, sub_data in data_dict.items():
        print('Checking %d-length images samples | Num:%d'%(n_img,len(sub_data)))

    preds = []
    data_dict = split_data(test_annotation['data'])
    try:
        for n_img, sub_data in data_dict.items():
            print('Proceeding %d-length images samples | Num:%d'%(n_img,len(sub_data)))
            E = EDataset(sub_data, test_annotation['metadata']['task_instruction'],img_dir)
            data_loader = torch.utils.data.DataLoader(dataset=E, batch_size=int(batch_size/n_img)+1,shuffle=False,num_workers = 8,collate_fn = collate_fn)

            for i,samples in enumerate(tqdm(data_loader)):
                pred_responses = chat.batch_answer(batch_raw_img_list=samples['raw_img_list'],batch_context=samples['context'])
                for sid, gt, p in zip(samples['sample_id'],samples['response'],pred_responses):
                    if torch.is_tensor(sid):
                        sid = sid.item()
                    preds.append({'sample_id':sid,'pred_response':p, 'gt_response':gt})

    finally:
        with open(os.path.join(output_dir,'pred.json'),'w',encoding='utf8') as f:
            preds.sort(key=lambda x: x["sample_id"])
            json.dump(preds,f,indent=4,ensure_ascii=False)

dataset_list = ["operation"]
for dataset in dataset_list:
    print(f"#############################\nInferring {dataset} dataset ...\n#############################\n")
    infer_dataset(dataset)

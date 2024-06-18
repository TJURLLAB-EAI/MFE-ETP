import os
import torch
import pickle
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import Dataset
import json
from tqdm import tqdm 
import sys
from PIL import Image
from torch.utils.data.dataloader import default_collate
import torch
from lavis.models import load_model_and_preprocess
from concurrent.futures import ThreadPoolExecutor
class EDataset(Dataset):
    def __init__(
        self, vis_processor, dataset, annoation, task_instructions, vis_root
    ):
        self.vis_root = vis_root
        self.annotation = annoation
        self.vis_processor = vis_processor
        self.task_instructions = task_instructions
        
    def __len__(self):
        return len(self.annotation)

    
    def __getitem__(self, index):
        images = []
        ann = self.annotation[index]
        task_instruction = self.task_instructions[ann['task_instruction_id']]
        prompt = task_instruction + ann['task_instance']['context']
        #remove img_symbol
        # for i in range(len(ann['task_instance']['images_path'])):
        #     rmv_txt = '{image#%d}'% (i+1)
        #     rmv_tbl = '{table#%d}'% (i+1)
        #     text_context = text_context.replace(rmv_txt, '')
        #     text_context = text_context.replace(rmv_tbl, '')
        for p in ann['task_instance']['images_path']:
            image = Image.open(os.path.join(self.vis_root, p)).convert("RGB")
            image = self.vis_processor(image)
            images.append(image)
        images = torch.stack(images, 1)
        return {
            "sample_id": ann['sample_id'],
            "image": images,
            "prompt": prompt,
            "response": str(ann['response'])
        }        
    def collater(self, samples):
        return default_collate(samples)

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
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        all_images_exist = all(executor.map(check_img_path, [p for ann in data for p in ann['task_instance']['images_path']]))

    if all_images_exist:
        print('All images exist')
    else:
        exit(-1)

def check_prompt(prompt):
    if len(prompt) < 6: # 10:
        print('No prompt')
        exit(-1)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="instructblipflant5xl")   # blip2flant5xl instructblipflant5xl
parser.add_argument('--output-root', type=str, default='./benchmark-evaluation')
parser.add_argument('--device', type=str, default="cuda:0")


args = parser.parse_args()

device = torch.device(args.device)
output_root = args.output_root
model_name = args.model

batch = 1

if model_name == 'blip2flant5xxl':
    name = 'blip2_t5'
    model_type="pretrain_flant5xxl"
elif model_name == 'blip2flant5xl':
    name = 'blip2_t5'
    model_type="pretrain_flant5xl"
elif model_name == 'instructblip7b':
    name="blip2_vicuna_instruct"
    model_type="vicuna7b"
elif model_name == 'instructblip13b':
    name="blip2_vicuna_instruct"
    model_type="vicuna13b"
elif model_name == 'instructblipflant5xxl':
    name='blip2_t5_instruct'
    model_type="flant5xxl"
elif model_name == 'instructblipflant5xl':
    name = 'blip2_t5_instruct'
    model_type = 'flant5xl'
else:
    print('model unsupported')
    exit(0)
print(name, model_type)
# device = torch.device(device) # if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True, device=device)


def infer_dataset(dataset):
    vis_root = './data/%s/images' % (dataset)
    print(vis_root)
    print(dataset)
    dataset_dir = './data'
    task_dir = os.path.join(dataset_dir, dataset)
    img_dir  = os.path.join(task_dir,'images')

    output_dir = os.path.join(output_root, model_name, dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(task_dir,'data.json')):
        print(f"no path {os.path.join(task_dir,'data.json')}")
        return

    test_annotation = json.load(open(os.path.join(task_dir,'data.json'),'r'))
    data_dict = split_data(test_annotation['data'])
    check_data(vis_root,test_annotation['data'])
    check_prompt(test_annotation['metadata']['task_instruction'])
    for n_img, sub_data in data_dict.items():
        print('Checking %d-length images samples | Num:%d'%(n_img,len(sub_data)))
    preds = []
    data_dict = split_data(test_annotation['data'])
    try:
        for n_img, sub_data in data_dict.items():
            print('Proceeding %d-length images samples | Num:%d'%(n_img,len(sub_data)))
            E = EDataset( vis_processors['eval'], dataset, sub_data, test_annotation['metadata']['task_instruction'],vis_root)
            data_loader = torch.utils.data.DataLoader(dataset=E, batch_size=int(batch/n_img)+1,shuffle=False,num_workers = 8)
            for i,samples in enumerate(tqdm(data_loader)):
                samples['image'] = samples['image'].to(device)
                pred_responses = model.generate(samples)
                for sid, gt, p in zip(samples['sample_id'],samples['response'],pred_responses):
                    preds.append({'sample_id':sid.item(),'pred_response':p, 'gt_response':gt})

    finally:
        with open(os.path.join(output_dir,'pred.json'),'w',encoding='utf8') as f:
            preds.sort(key=lambda x: x["sample_id"])
            json.dump(preds,f,indent=4,ensure_ascii=False)


dataset_list = ["object_type", "property", "time_perception", "spatial_perception", "relevant_object", "operation", "goal", "sequence"]
for dataset in dataset_list:
    print(f"#############################\nInferring {dataset} dataset ...\n#############################\n")
    infer_dataset(dataset)

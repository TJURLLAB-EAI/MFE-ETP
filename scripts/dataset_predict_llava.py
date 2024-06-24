import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
# import pandas as pd
import numpy as np
import argparse
from torch.utils.data import Dataset
import json
from tqdm import tqdm 
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria



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
            context = context.replace(rmv_txt, DEFAULT_IMAGE_TOKEN+'\n')
            context = context.replace(rmv_tbl, DEFAULT_IMAGE_TOKEN+'\n')
        conv = conv_templates["llava_v0"].copy()
        conv.append_message(conv.roles[0], context)
        conv.append_message(conv.roles[1], None)
        context = conv.get_prompt()
        for p in ann['task_instance']['images_path']:
            img_path = os.path.join(self.img_dir, p)
            raw_img = Image.open(img_path).convert('RGB').resize((224,224))
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

def batch_generate(model, image_processor, tokenizer, batch_img_list, batch_context, max_new_tokens=128,num_beams=1):
    device='cuda'
    batch_img_embd_list = []
    for img_list in batch_img_list:
        list_embd = image_processor.preprocess(img_list, return_tensors="pt")["pixel_values"]
        batch_img_embd_list.append(list_embd)
    batch_imbd = torch.stack(batch_img_embd_list, dim=0).to(device)
    input_ids = tokenizer(batch_context).input_ids
    max_prompt_size = max([len(input_id) for input_id in input_ids])
    for i in range(len(input_ids)):
        padding_size = max_prompt_size - len(input_ids[i])
        input_ids[i] = [tokenizer.pad_token_id] * padding_size + input_ids[i]
    input_ids = torch.as_tensor(input_ids).to(device)
    stop_str = "###"
    with torch.inference_mode():
        output_ids = model.generate(
                input_ids,
                images=batch_imbd.half().cuda(device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True)
        len_input = torch.as_tensor(input_ids).to(device).size()[1]
        output_ids = output_ids[:,len_input:]
        batch_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for i, outputs in enumerate(batch_outputs):
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            batch_outputs[i] = outputs
    return batch_outputs


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
parser.add_argument('--model', type=str, default="llava-v1.5-7b")
parser.add_argument('--gpuid', type=str, default="0")

args = parser.parse_args()

gpuid = args.gpuid

batch_size = args.batch


model_path = "liuhaotian./llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)
# print(model_name)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# dataset = args.dataset

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
            E = EDataset( sub_data, test_annotation['metadata']['task_instruction'],img_dir)
            data_loader = torch.utils.data.DataLoader(dataset=E, batch_size=int(batch_size/n_img)+1,shuffle=False,num_workers = 8,collate_fn = collate_fn)
            for i,samples in enumerate(tqdm(data_loader)):
                pred_responses = batch_generate(model, image_processor, tokenizer, samples['raw_img_list'], samples['context'])
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

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import random
import json
import time

import torch
import numpy as np
from tqdm import tqdm

import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide 

from PIL import Image  
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from sam_caf import hyper_params_tuning, get_crops, retrieve_relevant_crop, retrieve_relevant_crop_biomed, get_sam_prompts, sam_predicton, retrieve_relevant_crop_biomed_topk


# In[2]:


#config
class DictToObject:
    def __init__(self, dict_obj):
        for key, value in dict_obj.items():
            setattr(self, key, value)

config_dict = {
    "model_name" : "SAM",
    "model_type" : "vit_h",
    "source":    "False", 
    "refine" : "False",
    "pre_trained": "True", 
    "sam_ckpt":  "/data/aofei/LLM/SAM/sam_vit_h_4b8939.pth", 
    "clip_prompts": "./clip_prompts/abd_seg.json"
}

config = DictToObject(config_dict)

prompt_mode, mode = "crops", "sam_clip"

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    return image


# In[3]:


import os
# os.environ["TRANSFORMERS_CACHE"]="/data/aofei/huggingface_cache/transformers"
os.environ["HF_HOME"]="/data/aofei/huggingface_cache/transformers"
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

biomed_clip_model, biomed_preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device="cuda")
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
sam_checkpoint = config.sam_ckpt

sam = sam_model_registry[config.model_type](checkpoint=sam_checkpoint)
sam.to("cuda")
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

dice_scores = []
mask_generator, area = hyper_params_tuning(sam)


# In[4]:


def sam_generation(image_path):
    image = preprocess_image(image_path=image_path)
    with torch.no_grad():
        # if mode == "sam_clip":
        masks = mask_generator.generate(image)
        masks = [mask for mask in masks if mask["area"] < area] # area filtering based on area value from hyper-params tuning
        img_crops = get_crops(image, masks, prompt_mode)
        
    return masks, img_crops

def filter_sam_results(masks, img_crops):
    new_masks, new_img_crops = [], []
    for i in range(len(masks)):
        mask = masks[i]
        if mask['bbox'][0] == 0 or mask['bbox'][1] == 0:
            continue
        if mask['bbox'][2] <= 12 or mask['bbox'][3] <= 12:
            continue
        y_max, x_max = mask['bbox'][1] + mask['bbox'][3], mask['bbox'][0] + mask['bbox'][2]
        if y_max > 253 or x_max > 253:
            continue
        new_masks.append(mask)
        new_img_crops.append(img_crops[i])
    return new_masks, new_img_crops

def get_topk_similar(k, crop_scores):
    sorted_scores = sorted([(i, m) for (i,m) in enumerate(crop_scores)], key=lambda x: x[1], reverse=True)
    return sorted_scores[:k]

def get_compelete_contour(masks):
    width_list = []
    # need to consider the chest xray
    for i in masks:
        width, height = i['bbox'][2], i['bbox'][3]
        width_list.append(width)
    sorted_width = sorted([(i, m) for (i,m) in  enumerate(width_list)], key=lambda x: x[1], reverse=True)
    return sorted_width[0][1]

def judge_inner_boxes(bboxes):
    for bbox in bboxes:
        bbox


# In[5]:


masks_image_dict = dict()
masks_all_image_dict = dict()
crops_image_dict = dict()
def generate_segments(query, image_path):
    if masks_image_dict.__contains__(image_path):
        masks = masks_image_dict[image_path]
        img_crops = crops_image_dict[image_path]
    else:
        masks, img_crops = sam_generation(image_path=image_path)
        masks, img_crops = filter_sam_results(masks, img_crops)
        masks_image_dict[image_path] = masks
        crops_image_dict[image_path] = img_crops
    img_crops_filtered = img_crops
    prompts = {"query": [query]}
    max_indices, scores = retrieve_relevant_crop_biomed_topk(img_crops, prompts, biomed_clip_model, biomed_preprocess, config, tokenizer=tokenizer, topk=4)
    # topk_indices = get_topk_similar(3, scores["query"])
    # define a set of rules, firstly return top3
    # if there is no explicit organs to be used as query, then just use the whole segmentation
    # if the smaller boxes are in the bigger box, then use all of them but assign higher weights on smaller inner boxes
    bboxes = []
    segs = []
    # print(max_indices)
    if max_indices is not None:
        for i in max_indices["query"]:
            bboxes.append(masks[i]["bbox"])
            segs.append(masks[i]["segmentation"])
        return bboxes, segs, max_indices["query"]
    else:
        return bboxes, segs, []


def generate_all_segments(image_path):
    if masks_all_image_dict.__contains__(image_path):
        masks = masks_all_image_dict[image_path]
    else:
        masks, img_crops = sam_generation(image_path=image_path)
        masks, img_crops = filter_sam_results(masks, img_crops)
        masks_all_image_dict[image_path] = masks
    return masks


# In[6]:


import json
with open(r"/data/aofei/hallucination/CARES/OmniMedVQA/training.json", "r") as f:
    data = json.load(f)
len(data)
all_train_data = data


# In[7]:


all_train_data[0]


# In[8]:


all_train_data_en = all_train_data


# In[9]:


import copy
train_data_seg = copy.deepcopy(all_train_data_en)
len(train_data_seg)


# In[10]:


len(set([i['image'] for i in train_data_seg]))


# In[ ]:


# len(set([i['img_name'] for i in train_data]))


# In[16]:


# for i in range(len(train_rad_data)):
#failure case: 432-3 + 100
# from tqdm import tqdm
for i in tqdm(range(len(train_data_seg))):
# for i in tqdm(range(20)):
    data = train_data_seg[i]
    image_path = os.path.join('/data/aofei/hallucination/OmniMedVQA/VQA/raw/OmniMedVQA', data["image"])
    question = data["conversations"][0]["value"].replace("<image>\n", "")
    # question = question.split("The candidate Options are:")[0]
    # print(question)
    query = question

    bbox, segs, max_indices = [], [], []
    try:
        bbox, segs, max_indices = generate_segments(query, image_path)
    except:
        continue
    # print(bbox)
    data['bbox'] = bbox
    data['mask'] = segs
    data["bbox_indices"] = max_indices
    


# In[22]:


# np.sum(train_data_seg[0]['mask'][0])
np.sum(train_data_seg[0]['mask'][1].astype(int))


# In[17]:


s = 0
for i in train_data_seg:
    if not i.__contains__("mask"):
        s+= 1
s


# In[ ]:





# In[18]:


train_data_seg[3]


# In[ ]:





# In[19]:


#training2
new_train_data = []
segments_dict = dict()
for i in train_data_seg:
    template = dict()
    
    # template['answer_type'] = i['answer_type']
    template['image'] = i['image']
    template['id'] = i['id']
    template['conversations'] = []
    template['bboxes'] = []
    template['masks'] = []
    segments_dict[str(i['id'])] = []
    if i.__contains__("bbox"):
        template['bboxes'] = i["bbox"]
    if i.__contains__("mask"):
        segments_dict[str(i['id'])] = i["mask"]

    # if i.__contains__("mask"):
    #     json_ready_segments = [
    #         arr.astype(int).tolist() for arr in i["mask"]
    #     ]
    #     template['masks'] = json_ready_segments
    # template['text'] = i['question']

    # new_qa = {"from": "human", "value": i['question']}
    # new_qa2 = {"from": "gpt", "value": str(i['answer'])}
    template['conversations'] = i['conversations']
    new_train_data.append(template)

new_train_data[-6]


# In[20]:


new_train_data_top4 = []
for i in new_train_data:
    j = i.copy()
    j["bboxes"] = j["bboxes"][:4]
    new_train_data_top4.append(j)


# In[22]:


new_train_data_top4[100]


# In[23]:


# segments_dict['0']
ed = 0
for i in segments_dict:
    if len(segments_dict[i]) == 0:
        ed += 1
ed


# In[24]:


# save the masks to npz file

np.savez_compressed("/data/aofei/hallucination/CARES/OmniMedVQA/training_segments_top4.npz", **segments_dict)


# In[25]:


len(new_train_data)


# In[ ]:





# In[ ]:





# In[26]:


with open('/data/aofei/hallucination/CARES/OmniMedVQA/training_masks_top4.json', 'w') as json_file:
    json.dump(new_train_data_top4, json_file, indent=4)


# In[34]:





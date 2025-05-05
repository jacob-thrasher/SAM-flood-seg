import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from random import shuffle
from segment_anything import SamPredictor, sam_model_registry
from data import FloodSeg, Sen1Flood11
from transformers import SamProcessor, SamModel
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
from tifffile import imread


root = '/home/WVU-AD/jdt0025/Documents/data/WVFlood'

_id = '1'
dataset1 = 'floodseg'
dataset2 = 'wvflood'
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 3), dpi=300)
img = Image.open(f'experiments/{dataset1}_bbox_k1_42/img_{dataset2}_pretrained_{_id}.png')    
ax[0].imshow(img)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Image')

img = Image.open(f'experiments/{dataset1}_bbox_k1_42/gt_{dataset2}_pretrained_{_id}.png')    
ax[1].imshow(img)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Ground Truth')

plt.tight_layout()
plt.savefig(f'figures/{dataset2}_sample_{_id}.png')

# _id = '1'

# dataset1 = 'floodseg'
# dataset2 = 'wvflood'
# fig, ax = plt.subplots(nrows=2, ncols=8, figsize=(10, 3), dpi=300)
# img = Image.open(f'experiments/{dataset1}_bbox_k1_42/pred_{dataset2}_pretrained_{_id}.png')    
# ax[0, 0].imshow(img)
# ax[0, 0].set_xticks([])
# ax[0, 0].set_yticks([])
# ax[0, 0].set_title('SAM (bbox)')
# ax[0, 0].set_ylabel('FAS -> WV')
# img = Image.open(f'experiments/{dataset1}_bbox_k1_42/pred_{dataset2}_finetuned_{_id}.png')    
# ax[0, 1].imshow(img)
# ax[0, 1].set_xticks([])
# ax[0, 1].set_yticks([])
# ax[0, 1].set_title('FT-SAM (bbox)')
# img = Image.open(f'experiments/{dataset1}_point_k1_42/pred_{dataset2}_pretrained_{_id}.png')    
# ax[0, 2].imshow(img)
# ax[0, 2].set_xticks([])
# ax[0, 2].set_yticks([])
# ax[0, 2].set_title('SAM (1pt)')
# img = Image.open(f'experiments/{dataset1}_point_k1_42/pred_{dataset2}_finetuned_{_id}.png')    
# ax[0, 3].imshow(img)
# ax[0, 3].set_xticks([])
# ax[0, 3].set_yticks([])
# ax[0, 3].set_title('FT-SAM (1pt)')

# img = Image.open(f'experiments/{dataset1}_point_k3_42/pred_{dataset2}_pretrained_{_id}.png')    
# ax[0, 4].imshow(img)
# ax[0, 4].set_xticks([])
# ax[0, 4].set_yticks([])
# ax[0, 4].set_title('SAM (3pt)')
# img = Image.open(f'experiments/{dataset1}_point_k3_42/pred_{dataset2}_finetuned_{_id}.png')    
# ax[0, 5].imshow(img)
# ax[0, 5].set_xticks([])
# ax[0, 5].set_yticks([])
# ax[0, 5].set_title('FT-SAM (3pt)')
# img = Image.open(f'experiments/{dataset1}_each_k3_42/pred_{dataset2}_pretrained_{_id}.png')    
# ax[0, 6].imshow(img)
# ax[0, 6].set_xticks([])
# ax[0, 6].set_yticks([])
# ax[0, 6].set_title('SAM (each)')
# img = Image.open(f'experiments/{dataset1}_each_k3_42/pred_{dataset2}_finetuned_{_id}.png')    
# ax[0, 7].imshow(img)
# ax[0, 7].set_xticks([])
# ax[0, 7].set_yticks([])
# ax[0, 7].set_title('FT-SAM (each)')


# dataset1 = 'sen1flood11'
# dataset2 = 'wvflood'
# img = Image.open(f'experiments/{dataset1}_bbox_k1_42/pred_{dataset2}_pretrained_{_id}.png')    
# ax[1, 0].imshow(img)
# ax[1, 0].set_xticks([])
# ax[1, 0].set_yticks([])
# ax[1, 0].set_ylabel('Sen1 -> WV')
# img = Image.open(f'experiments/{dataset1}_bbox_k1_42/pred_{dataset2}_finetuned_{_id}.png')    
# ax[1, 1].imshow(img)
# ax[1, 1].set_xticks([])
# ax[1, 1].set_yticks([])
# img = Image.open(f'experiments/{dataset1}_point_k1_42/pred_{dataset2}_pretrained_{_id}.png')    
# ax[1, 2].imshow(img)
# ax[1, 2].set_xticks([])
# ax[1, 2].set_yticks([])
# img = Image.open(f'experiments/{dataset1}_point_k1_42/pred_{dataset2}_finetuned_{_id}.png')    
# ax[1, 3].imshow(img)
# ax[1, 3].set_xticks([])
# ax[1, 3].set_yticks([])

# img = Image.open(f'experiments/{dataset1}_point_k3_42/pred_{dataset2}_pretrained_{_id}.png')    
# ax[1, 4].imshow(img)
# ax[1, 4].set_xticks([])
# ax[1, 4].set_yticks([])
# img = Image.open(f'experiments/{dataset1}_point_k3_42/pred_{dataset2}_finetuned_{_id}.png')    
# ax[1, 5].imshow(img)
# ax[1, 5].set_xticks([])
# ax[1, 5].set_yticks([])
# img = Image.open(f'experiments/{dataset1}_each_k3_42/pred_{dataset2}_pretrained_{_id}.png')    
# ax[1, 6].imshow(img)
# ax[1, 6].set_xticks([])
# ax[1, 6].set_yticks([])
# img = Image.open(f'experiments/{dataset1}_each_k3_42/pred_{dataset2}_finetuned_{_id}.png')    
# ax[1, 7].imshow(img)
# ax[1, 7].set_xticks([])
# ax[1, 7].set_yticks([])

# plt.tight_layout()
# plt.savefig(f'figures/{dataset2}_{_id}')

# dataset = Sen1Flood11(root, os.path.join(root, 'train.csv'), None, region_select='point', k=1)
# img, mask = dataset[2]

# print(img.shape)
# print(mask.shape)



# fig, ax = plt.subplots(nrows=1, ncols=2)

# img = torch.tensor(img)
# ax[0].imshow(img.permute(1, 2, 0))
# ax[1].imshow(mask, alpha=1)
# plt.show()


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
# model = SamModel.from_pretrained('facebook/sam-vit-base')

# train_dataset = FloodSeg(os.path.join(root, 'Train/Image'), os.path.join(root, 'Train/Mask'), processor, region_select='each', k=1)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


# inputs = train_dataset[2]
# # bbox = inputs['input_boxes'][0]
# # bbox = [x // 4 for x in bbox]
# # print(inputs['pixel_values'].size())
# # print(inputs['ground_truth_mask'].shape)

# points = inputs['input_points']
# # points = [x // 4 for x in points]

# fig, ax = plt.subplots(nrows=1, ncols=2)
# A = inputs['pixel_values']
# A -= A.min(1, keepdim=True)[0]
# A /= A.max(1, keepdim=True)[0]
# ax[0].imshow(A.permute(1, 2, 0))
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# ax[1].imshow(inputs['ground_truth_mask'], alpha=1)

# # rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], edgecolor='red', linewidth=2, facecolor='none')
# # ax[1].add_patch(rect)

# points = points.squeeze(0)
# print(points)
# x = points[:, 0] // 4
# y = points[:, 1] // 4
# print(x)
# print(y)

# ax[1].scatter(x, y, color='red', marker='^', s=100)


# ax[1].set_xticks([])
# ax[1].set_yticks([])
# plt.show()
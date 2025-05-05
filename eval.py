import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from segment_anything import SamPredictor, sam_model_registry
from transformers import SamProcessor, SamModel
from torch.optim import Adam
from statistics import mean
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
import monai
import random
from data import FloodSeg, Sen1Flood11, WVFlood
from torchmetrics.functional.classification import binary_jaccard_index


def get_metrics(model, dataloader):

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    model.eval()
    running_dice = 0
    running_iou = 0
    with torch.no_grad():
        for batch in dataloader:

            # Configure inputs
            inputs = {k: v.to(device) for k, v in batch.items() if k != "ground_truth_mask"}

            # Get masks
            outputs = model(**inputs, multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)

            # Calculate loss
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            running_dice += (1 - loss)



            running_iou += binary_jaccard_index(F.sigmoid(predicted_masks).squeeze(), ground_truth_masks)

    dice = running_dice / len(dataloader)
    iou = running_iou / len(dataloader)

    # print(f'Dice: {dice}')
    # print(f'IOU : {iou}')

    return dice.item(), iou.item()


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='', required=True)
parser.add_argument("--do_finetune_weights", type=int, default=0, required=False)
parser.add_argument("--output_name", type=str, default='eval', required=False)
arguments = parser.parse_args()




# Eval Config
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)  
random.seed(seed)

csv_name = f'{arguments.output_name}.csv'
if not os.path.exists(csv_name):
    f = open(csv_name, 'w')
    writer = csv.writer(f)
    header = ['exp_id', 'finetuned', 'dice', 'iou']
    writer.writerow(header)
else:
    f = open(csv_name, 'a')
    writer = csv.writer(f)


do_finetune_weights = arguments.do_finetune_weights
folder = arguments.model


components = folder.split('_')
# dataset = components[0]
dataset = 'wvflood'
region_method = components[1]
k = int(components[2][1])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
finetune_weights = f'experiments/{folder}/best_model.pth'

if do_finetune_weights == 0:
    print(f"Evaluating model {folder} with PRETRAINED weights")
elif do_finetune_weights == 1:
    print(f"Evaluating model {folder} with FINETUNED weights")


# Load processor and model
## Processor preprocesses images in the way that SAM expects them to be
processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
model = SamModel.from_pretrained('facebook/sam-vit-base')

if do_finetune_weights == 1:
    print('Loading finetuned weights...')
    model.load_state_dict(torch.load(finetune_weights, weights_only=True))
model.to('cuda')
model.eval()

# Define dataloaders
if dataset == 'floodseg':
    root = '/home/WVU-AD/jdt0025/Documents/data/Flood'
    # train_dataset = FloodSeg(root, os.path.join(root, 'train.csv'), processor, region_select=region_method, k=k)
    test_dataset = FloodSeg(root, os.path.join(root, 'test.csv'), processor, region_select=region_method, k=k)
if dataset == 'wvflood':
    root = '/home/WVU-AD/jdt0025/Documents/data/WVFlood'
    # train_dataset = FloodSeg(root, os.path.join(root, 'train.csv'), processor, region_select=region_method, k=k)
    test_dataset = WVFlood(root, processor, region_select=region_method, k=k)
elif dataset == 'sen1flood11':
    root = '/home/WVU-AD/jdt0025/Documents/data/v1.1/data/flood_events/HandLabeled'
    # train_dataset = Sen1Flood11(root, os.path.join(root, 'train_clean.csv'), processor, region_select=region_method, k=k)
    test_dataset  = Sen1Flood11(root, os.path.join(root, 'test_clean.csv'), processor, region_select=region_method, k=k)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=True)


# Quantiative metrics
dice, iou = get_metrics(model, test_dataloader)
writer.writerow([folder, do_finetune_weights == 1, dice, iou])


# Get segmentation masks
batch = next(iter(test_dataloader))
inputs = {k: v.to(device) for k, v in batch.items() if k != "ground_truth_mask"}
outputs = model(**inputs, multimask_output=False)
pred = outputs.pred_masks.squeeze().detach().cpu()


for i in range(len(pred)):
    # Get test image and mask
    img = batch['pixel_values'][i].detach().cpu()
    mask = batch['ground_truth_mask'][i].detach().cpu()
    resize = T.Resize((256, 256))


    # Normalize image but I don't remember why lol
    A = img
    A -= A.min(1, keepdim=True)[0]
    A /= A.max(1, keepdim=True)[0]
    img = resize(A)
    mask = resize(mask)


    prefix = 'finetuned' if do_finetune_weights == 1 else 'pretrained'
    save_image(img, os.path.join('experiments', folder, f'img_{dataset}_{prefix}_{str(i)}.png'))

    if len(mask.size()) == 2: mask = mask.unsqueeze(0)
    save_image(mask, os.path.join('experiments', folder, f'gt_{dataset}_{prefix}_{str(i)}.png'))


    # Convert unbounded predictions into probabilities (sigmoid)
    # Round probabilities to nearest value (0, 1)
    pred_img = torch.round(F.sigmoid(pred[i]))


    save_image(pred_img, os.path.join('experiments', folder, f'pred_{dataset}_{prefix}_{str(i)}.png'))


f.close()




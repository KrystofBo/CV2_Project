from mmseg.apis import init_model, inference_model
import mmcv
import os
import numpy as np
from PIL import Image
import torch

config_file = 'mmsegmentation/configs/segmenter/segmenter_vit-b_mask_8xb1-160k_ade20k-512x512.py'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')

img_dir = 'gaussian-grouping/data/bear/images'
output_dir = 'gaussian-grouping/data/bear/object_mask_segmenter'
os.makedirs(output_dir, exist_ok=True)

for img_name in sorted(os.listdir(img_dir)):
    if not img_name.endswith(('.jpg', '.png')):
        continue
    img_path = os.path.join(img_dir, img_name)
    result = inference_model(model, img_path)
    mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

    # Save as .png mask
    Image.fromarray(mask).save(os.path.join(output_dir, img_name.replace('.jpg', '.png')))

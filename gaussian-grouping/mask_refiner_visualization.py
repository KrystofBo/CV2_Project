import os
import os.path as osp
import shutil
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import colorsys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

def id2rgb(id, max_num_obj=256):

    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5
    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)

    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.ones((*objects.shape[-2:], 3), dtype=np.uint8) * 255
    all_obj_ids = np.unique(objects)

    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask

    return rgb_mask

device = 'cuda'
sam = sam_model_registry['vit_h'](checkpoint='/home/scur0695/gaussian-grouping/sam_checkpoint/sam_vit_h_4b8939.pth')
sam.to(device=device)
predictor = SamPredictor(sam)
image = cv2.imread('/home/scur0695/gaussian-grouping/data/room0/images/rgb_0.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# Load ID mask
img_id = cv2.imread('/home/scur0695/gaussian-grouping/output/room0_uncertain/train/ours_30000/objects_id/rgb_90.png', cv2.IMREAD_UNCHANGED)
if img_id.ndim == 3:
    img_id = img_id[:, :, 0]  # Use red channel if RGB

new_id_mask = np.ones_like(img_id, dtype=np.uint8) * 255
unique_ids = np.unique(img_id)
start_idx = 0
obj_ids = unique_ids[start_idx:]
rgb_mask = visualize_obj(img_id)
for obj_id in obj_ids:

    mask = (img_id == obj_id).astype(np.uint8)

    # Find all connected components (disconnected regions of the same obj_id)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    for contour in contours:
        # Skip tiny noise less than 1 percent of the image size
        if cv2.contourArea(contour) < 0.0025 * image.shape[0] * image.shape[1]:
            continue


        x, y, w, h = cv2.boundingRect(contour)
        input_box = np.array([[x, y], [x + w, y + h]])

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1.5)
        cv2.rectangle(rgb_mask, (x, y), (x + w, y + h), (0, 255, 0), 1.5)
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=False
        )

        if masks is not None and len(masks) > 0:
            sam_mask = masks[0]
            new_id_mask[sam_mask] = obj_id

# Save new mask
cv2.imwrite('img_visualization.png', image)
cv2.imwrite('mask_visualization.png', cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))

import numpy as np
import cv2
import colorsys
import matplotlib.pyplot as plt
from PIL import Image

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


scannet_mapping = {
    0: 0, #wall
    26: 0, #shower wall
    2: 1, #floor
    1: 2, #ceiling
    4: 3, #door
    19: 3, #doorframe
    6: 4, #cabinet
    22: 4, #kitchen cabinet
    8: 5, #curtain
    18: 6, #shelf
    12: 6, #bookshelf
    3: 7, #table
    11: 8, #Office chair
    9: 8, #chair
    56: 9, #bottle
    37: 10, #sink
    25: 11, #bed
    17: 12, #monitor
    52: 13, #microwave
    14: 14, #window
    16: 14, #window frame
    15: 15, #box
    40: 16, #pillow
    29: 17, #plant
    7: 18, #blinds
    27: 19, #trash can
}

ade20k_mapping = {
    0:   0,   # Wall
    3:   1,   # Floor
    5:   2,   # Ceiling
    14:  3,   # Door
    10:  4,   # Cabinet
    18:  5,   # Curtain
    24:  6,   # Shelf
    15:  7,   # Table
    56:  7,   # Pool table
    64:  7,   # Coffee table
    33:  7,   # Desk
    19:  8,   # Chair
    30:  8,   # Armchair
    75:  8,   # Swivel chair
    98:  9,   # Bottle
    47:  10,  # Sink
    7:   11,  # Bed
    74:  12,  # Monitor
    124:  13,  # Microwave
    8:   14,  # Window
    41:  15,  # Box
    57:  16,  # Pillow
    17:  17,  # Plant
    63:  18,  # Blinds
    138: 19,  # Trash can
}
img_path = '/home/scur0695/gaussian-grouping/output/scannet4_uncertain_refine/test/ours_30000/objects_id/DSC03913.png'
img_name = img_path.split('/')[-1]
img = Image.open(img_path)
img = np.array(img)
for k, v in ade20k_mapping.items():
    img[img == k] = v

img = visualize_obj(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(f'{img_name}', img)
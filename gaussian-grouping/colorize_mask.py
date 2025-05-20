import os
import argparse
import numpy as np
from PIL import Image
import colorsys

# Color mapping function (copied from render.py)
def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)
    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

def colorize_and_save(mask_path):
    # Load mask (should be single-channel, grayscale)
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]  # Take first channel if accidentally RGB
    colored_mask = visualize_obj(mask)
    # Prepare output path
    folder, fname = os.path.split(mask_path)
    out_folder = folder + '_colored'
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, fname)
    Image.fromarray(colored_mask).save(out_path)
    print(f"Saved colored mask to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Colorize black and white mask images using the same color mapping as Gaussian Grouping.")
    parser.add_argument('mask_paths', nargs='+', help='Paths to mask images to colorize')
    args = parser.parse_args()
    for mask_path in args.mask_paths:
        colorize_and_save(mask_path)

if __name__ == '__main__':
    main() 
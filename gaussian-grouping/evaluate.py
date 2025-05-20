from PIL import Image
import torch
import numpy as np
import os
import sys
import json

'''
To run this script, you need to prepare a subset of classes that has overlap between ADE20K and Scannet++ or Replica you want to evaluate.
- Semantic classes: a dictionary mapping class ids of the subset to the class names of the subset
- Scannet mapping: a dictionary mapping Scannet ids to the class ids of the subset
- ADE20K mapping: a dictionary mapping ADE20K ids to the class ids of the subset.
'''

# Scannet dataset classes
scannet_semantic_classes = [
    "wall", "floor", "ceiling", "door", "cabinet", "curtain", "shelf",
    "table", "chair", "bottle", "sink", "bed", "monitor", "microwave",
    "window", "box", "pillow", "plant", "blinds", "trash can"
]

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

# Replica dataset classes
replica_semantic_classes = [
    'wall', 'ceiling', 'blinds', 'sofa', 'rug', 'lamp', 'floor', 'pillow', 
    'chair', 'door', 'pillar', 'table', 'monitor', 'blanket', 'tv-screen', 
    'picture', 'cushion', 'bed', 'desk'
]

replica_mapping = {
    93: 0,  # wall
    31: 1,  # ceiling
    12: 2,  # blinds
    76: 3,  # sofa
    98: 4,  # rug
    47: 5,  # lamp
    40: 6,  # floor
    61: 7,  # pillow
    20: 8,  # chair
    37: 9,  # door
    60: 10, # pillar
    80: 11, # table
    52: 12, # monitor
    11: 13, # blanket
    87: 14, # tv-screen
    59: 15, # picture
    29: 16, # cushion
    7: 17,  # bed
    34: 18, # desk
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

# ADE20K mapping for replica dataset 
replica_ade20k_mapping = {
    0:   0,   # Wall
    5:   1,   # Ceiling
    63:  2,   # Blinds
    23:  3,   # Sofa
    28: 4,   # Rug
    36:  5,   # Lamp
    3:   6,   # Floor
    57:  7,   # Pillow
    19:  8,   # Chair
    14:  9,   # Door
    42: 10,  # Pillar 
    15:  11,  # Table
    143:  12,  # Monitor
    131: 13,  # Blanket
    89:  14,  # TV-screen
    22:  15,  # Picture
    39:  16,  # Cushion
    7:   17,  # Bed
    33:  18,  # Desk
}

def IoU(output, target, c, ignore_index=255):
    '''
    This function computes the intersection and union of two tensors.
    Args:
        output: The predicted segmentation map (tensor of shape [H, W])
        target: The ground truth segmentation map (tensor of shape [H, W])
        c: The number of classes
        ignore_index: The index to ignore in the evaluation (default is 255)
    '''
    if output.shape == target.shape:
        pass
    else:
        output = Image.fromarray(output.cpu().numpy().astype(np.uint8))
        output = output.resize((target.shape[1], target.shape[0]), Image.NEAREST)
        output = np.array(output)
        output = torch.from_numpy(output)
    
    assert output.shape == target.shape
    output = output
    target = target
    mask = (target != ignore_index)
    output = output[mask]
    target = target[mask]
    intersect = output[output == target]
    area_intersect = torch.histc(
        intersect.float(), bins=(c), min=0, max=c - 1)
    area_pred_label = torch.histc(
        output.float(), bins=(c), min=0, max=c - 1)
    area_label = torch.histc(
        target.float(), bins=(c), min=0, max=c - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union

def preprocess_image(image, mapppings):
    '''
    This function preprocesses the input image by mapping the class ids to the corresponding class ids in the subset.
    '''
    label_map = torch.ones(image.shape[0], image.shape[1], dtype=torch.int64) * 255 # Choose 255 is the ignore index
    for k, v in mapppings.items():
        label_map[image == k] = v
    return label_map

def evaluate(outputs, targets, num_classes, ignore_index=255):
    '''
    This function evaluates the model by computing the intersection and union of the predicted and ground truth segmentation maps.
    Args:
        outputs: A list of predicted segmentation maps (each of shape [H, W])
        targets: A list of ground truth segmentation maps (each of shape [H, W])
        num_classes: The number of classes in the dataset
        ignore_index: The index to ignore in the evaluation (default is 255)
    '''
    assert len(outputs) == len(targets)
    total_intersect = torch.zeros(num_classes)
    total_union = torch.zeros(num_classes)
    for output, target in zip(outputs, targets):
        intersect, union = IoU(output, target, num_classes, ignore_index)
        total_intersect += intersect
        total_union += union
    return total_intersect, total_union

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python evaluate.py <output_base_dir> <target_base_dir> <dataset> <results_path> <scene_list>")
        print("dataset: scannet or replica")
        print("scene_list: comma-separated list of scene names (e.g., office0,office1,room0)")
        sys.exit(1)

    output_base_dir = sys.argv[1]  # Base directory containing scene output dirs
    target_base_dir = sys.argv[2]  # Base directory containing scene ground truth dirs
    dataset = sys.argv[3].lower()  # Dataset type: 'scannet' or 'replica'
    results_path = sys.argv[4]  # Path to save the results
    scene_list = sys.argv[5].split(",") if len(sys.argv) > 5 else [] # List of scenes to evaluate
    
    if dataset not in ['scannet', 'replica']:
        print("Dataset must be either 'scannet' or 'replica'")
        sys.exit(1)

    # Set the appropriate mappings and semantic classes based on the dataset
    if dataset == 'scannet':
        target_mapping = scannet_mapping
        semantic_classes = scannet_semantic_classes
        ade_mapping = ade20k_mapping
    else:  # replica
        target_mapping = replica_mapping
        semantic_classes = replica_semantic_classes
        ade_mapping = replica_ade20k_mapping

    # For scannet, use the original path style without scene subfolders
    if dataset == 'scannet':
        # Collect all outputs and targets directly from the specified directories
        all_outputs = []
        all_targets = []
        file_count = 0
        
        output_files = sorted(os.listdir(output_base_dir))
        
        for output_file in output_files:
            # Read the output ids map
            output_path = os.path.join(output_base_dir, output_file)
            output = np.array(Image.open(output_path))
            output = preprocess_image(output, ade_mapping)

            # Read the target ids map
            target_file = output_file[:-4] + '.JPG.png'
            target_path = os.path.join(target_base_dir, target_file)
            
            if os.path.exists(target_path):
                target = np.array(Image.open(target_path))
                target = preprocess_image(target, target_mapping)

                all_outputs.append(output)
                all_targets.append(target)
                file_count += 1
            else:
                print(f"Warning: Target file {target_path} not found, skipping {output_file}")
        
        print(f"Total files processed: {file_count}")
        scene_file_counts = {"scannet": file_count}  # Single scene count
        
    else:  # For replica, use the nested directory structure
        # Find all scene directories if not specified
        if not scene_list:
            scene_list = [d for d in os.listdir(output_base_dir) 
                        if os.path.isdir(os.path.join(output_base_dir, d))]
            scene_list = [d for d in scene_list 
                        if os.path.isdir(os.path.join(target_base_dir, d))]
            
        if not scene_list:
            print(f"No matching scene directories found in {output_base_dir} and {target_base_dir}")
            sys.exit(1)
            
        print(f"Evaluating scenes: {', '.join(scene_list)}")
        
        # Collect all outputs and targets from all scenes into single lists
        all_outputs = []
        all_targets = []
        file_count = 0
        scene_file_counts = {}
        
        # Process each scene and collect all files
        for scene in scene_list:
            # Full paths to the actual directories containing the images
            output_dir = os.path.join(output_base_dir, scene, "test", "ours_30000", "objects_id")
            target_dir = os.path.join(target_base_dir, scene, "semantic_class")
            
            if not os.path.isdir(output_dir):
                print(f"Warning: Output directory {output_dir} not found, skipping scene {scene}")
                continue
                
            if not os.path.isdir(target_dir):
                print(f"Warning: Target directory {target_dir} not found, skipping scene {scene}")
                continue
                
            print(f"Processing scene: {scene}")
            
            output_files = sorted(os.listdir(output_dir))
            scene_file_count = 0
            
            for output_file in output_files:
                # Read the output ids map
                output_path = os.path.join(output_dir, output_file)
                output = np.array(Image.open(output_path))
                output = preprocess_image(output, ade_mapping)

                # Read the target ids map
                target_file = 'semantic_class_' + output_file.split('_')[1]
                target_path = os.path.join(target_dir, target_file)
                
                if os.path.exists(target_path):
                    target = np.array(Image.open(target_path))
                    target = preprocess_image(target, target_mapping)

                    all_outputs.append(output)
                    all_targets.append(target)
                    scene_file_count += 1
                else:
                    print(f"Warning: Target file {target_path} not found, skipping {output_file}")
            
            print(f"Added {scene_file_count} files from scene {scene}")
            scene_file_counts[scene] = scene_file_count
            file_count += scene_file_count
    
    if file_count == 0:
        print("No valid files found for evaluation")
        sys.exit(1)
    
    # Evaluate all files together
    total_intersect, total_union = evaluate(all_outputs, all_targets, len(semantic_classes), ignore_index=255)
    
    # Calculate IoUs
    ious = {}
    valid_classes = 0
    
    for i in range(len(semantic_classes)):
        if total_union[i] == 0:
            continue
        iou = total_intersect[i] / total_union[i]
        ious[semantic_classes[i]] = float(iou)
        valid_classes += 1
        print(f'IoU for class {semantic_classes[i]}: {iou:.4f}')
    
    if valid_classes > 0:
        miou = sum(ious.values()) / valid_classes
        ious['mIoU'] = float(miou)
        print(f'mIoU: {miou:.4f}')
    else:
        print("No valid classes found across all scenes")
    
    # Store results
    final_results = {
        "metrics": ious,
        "scene_file_counts": scene_file_counts,
        "total_files": file_count
    }
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
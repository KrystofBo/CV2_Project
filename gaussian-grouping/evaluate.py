from PIL import Image
import torch
import numpy as np
import os
import sys

'''
To run this script, you need to prepare a subset of classes that has overlap between ADE20K and Scannet++ or Replica you want to evaluate.
- Semantic classes: a dictionary mapping class ids of the subset to the class names of the subset
- Scannet mapping: a dictionary mapping Scannet ids to the class ids of the subset
- ADE20K mapping: a dictionary mapping ADE20K ids to the class ids of the subset.
'''



semantic_classes = ['wall', 'floor', 'plant', 'ceiling', 'bed', 'window', 'cabinet', 'door', 'table', 'curtain', 'chair', 'painting', 'sofa', 'shelf', 'storage cabinet', 'table lamp', 'cushion', 'rack', 'box', 'poster', 'kitchen counter', 'sink', 'refrigerator', 'pillow', 'bookshelf', 'blinds', 'toilet', 'book', 'computer tower', 'office chair', 'towel', 'ceiling lamp', 'tv', 'jacket', 'bottle', 'basket', 'bag', 'microwave', 'plant pot', 'blanket', 'exhaust fan', 'container', 'trash can', 'monitor', 'whiteboard', 'heater', 'cup', 'clock']
semantic_classes = {i: name for i, name in enumerate(semantic_classes)}
idxs_mapping = {1: [1], 3: [4], 30: [5, 10, 18, 67, 73], 2: [6], 26: [8], 15: [9], 7: [11, 45, 56], 5: [15, 59], 4: [16, 34, 57, 65], 9: [19], 10: [20, 31, 32, 70, 111], 52: [23], 24: [24], 19: [25], 11: [36], 73: [37], 59: [40, 98], 47: [41], 16: [42], 51: [44, 101], 34: [46, 71, 74, 100], 38: [48], 35: [51], 41: [58], 13: [63], 8: [64, 87], 48: [66], 29: [68], 33: [75], 12: [76], 42: [82], 6: [83, 86, 135], 32: [90], 36: [93], 57: [99], 60: [113], 39: [116], 53: [125], 67: [126, 136], 31: [132], 68: [134, 140], 78: [138], 28: [139], 18: [142, 144], 14: [145], 22: [147], 69: [148], 87: [149]}
scannet_mapping = {k: i for i, k in enumerate(idxs_mapping.keys())}
ade20k_mapping = {j: i for i, v in enumerate(idxs_mapping.values()) for j in v}


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
        output = torch.from_numpy(output).cuda()
    
    assert output.shape == target.shape
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
    total_intersect = torch.zeros(num_classes).cuda()
    total_union = torch.zeros(num_classes).cuda()
    for output, target in zip(outputs, targets):
        intersect, union = IoU(output, target, num_classes, ignore_index)
        total_intersect += intersect
        total_union += union
    return total_intersect, total_union

outputs = []
targets = []

output_path = sys.argv[1] # This path contains the path contain model outputs (in ids map format)
target_path = sys.argv[2] # This path contains the path contain ground truth labels (in ids map format)

output_files = sorted(os.listdir(output_path))
target_files = sorted(os.listdir(target_path))

for output_file in output_files:
    # Read the output ids map
    output = np.array(Image.open(os.path.join(output_path, output_file)))
    output = preprocess_image(output, ade20k_mapping).cuda()

    # Read the target ids map
    target_file = output_file[:-4] +'.JPG.png' # Replace this with how you name your target files
    target = np.array(Image.open(os.path.join(target_path, target_file)))
    target = preprocess_image(target, scannet_mapping).cuda()

    outputs.append(output)
    targets.append(target)


total_intersect, total_union = evaluate(outputs, targets, len(semantic_classes), ignore_index=255)
ious = []
for i in range(len(semantic_classes)):
    if total_union[i] == 0:
        continue
    iou = total_intersect[i] / total_union[i]
    print(f'IoU for class {semantic_classes[i]}: {iou:.4f}')
    ious.append(iou)
mIoU = sum(ious) / len(ious)
print(f'mIoU: {mIoU:.4f}')

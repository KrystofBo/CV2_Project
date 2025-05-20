import os
import os.path as osp
import numpy as np
import cv2 
import shutil
import argparse
from mmseg.apis import MMSegInferencer


def run_inference_and_save_results(
    checkpoint_file: str,
    image_paths: list[str],
    output_base_dir: str, 
    device: str = 'cuda:0' 
):
    """
    Performs semantic segmentation inference using a pretrained MMSegmentation model
    and saves the label maps and colormap visualizations using Inferencer.

    Args:
        config_file (str): Path to the model configuration file.
        checkpoint_file (str): Path to the pretrained model checkpoint file.
        image_paths (list[str]): A list of paths to the input images.
        output_base_dir (str): Base directory to save results. Inferencer will
                                create 'preds' (label maps) and 'vis' (colormaps)
                                subdirectories within this.
        device (str): Device to use for inference (e.g., 'cuda:0', 'cpu').
    """

    inferencer = MMSegInferencer(
        model=checkpoint_file,
        device=device,
    )
    print("Inferencer initialized successfully.")

    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Output results will be saved under: {output_base_dir}")
    print("Inferencer will create 'preds' (label maps) and 'vis' (colormap visualizations) subdirectories.")


    print(f"\nStarting inference on {len(image_paths)} images...")
    for image_path in image_paths:
        img_name = os.path.basename(image_path).split('.')[0]
        print(f"Processing image: {img_name}", flush=True)
        result = inferencer(
            inputs=image_path,
            show=False,
            return_datasamples=True
        )
        pred_dir = output_base_dir
        
        pred_mask = result.pred_sem_seg.data[0]
        logits = result.seg_logits.data

        logits = logits.softmax(dim=0)
        confidence_score = logits.max(dim=0)[0]
        null_map = confidence_score < 0.9
        pred_mask[null_map] = 255
        pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
        # Save prediction mask
        cv2.imwrite(os.path.join(pred_dir, f"{img_name}.png"), pred_mask.astype(np.uint8))
            
    print("\nInference and saving complete.")


def create_uncertain_folder(folder_name):
    """
    Creates a copy of the given folder with '_uncertain' suffix and clears the object_mask directory.
    
    Args:
        folder_name (str): Name of the data folder (e.g., 'office0', 'room0')
    
    Returns:
        str: Name of the created folder
    """
    source_folder = osp.join('data', folder_name)
    uncertain_folder_name = f"{folder_name}_uncertain"
    target_folder = osp.join('data', uncertain_folder_name)
    
    # Check if source folder exists
    if not osp.exists(source_folder):
        raise ValueError(f"Source folder {source_folder} does not exist")
    
    # Create a copy of the folder
    if osp.exists(target_folder):
        print(f"Target folder {target_folder} already exists, removing it first")
        shutil.rmtree(target_folder)
    
    print(f"Copying {source_folder} to {target_folder}")
    shutil.copytree(source_folder, target_folder)
    
    # Clear the object_mask directory but leave the folder
    object_mask_dir = osp.join(target_folder, 'object_mask')
    if osp.exists(object_mask_dir):
        print(f"Clearing contents of {object_mask_dir}")
        for item in os.listdir(object_mask_dir):
            item_path = osp.join(object_mask_dir, item)
            if osp.isfile(item_path):
                os.remove(item_path)
    else:
        print(f"Creating {object_mask_dir}")
        os.makedirs(object_mask_dir, exist_ok=True)
    
    return uncertain_folder_name


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run MMSegmentation inference on a data folder')
    parser.add_argument('folder_name', type=str, help='Name of the data folder (e.g., office0, room0)')
    args = parser.parse_args()
    
    # Create uncertain folder
    uncertain_folder_name = create_uncertain_folder(args.folder_name)
    
    # Set up paths
    CHECKPOINT_FILE = osp.join('segformer_mit-b5_8xb2-160k_ade20k-512x512')
    
    input_images_dir = osp.join('data', uncertain_folder_name, 'images')
    list_of_image_paths = [
        osp.join(input_images_dir, img_name) for img_name in os.listdir(input_images_dir)
    ]
    
    output_base_directory = osp.join('data', uncertain_folder_name, 'object_mask')
    
    # Run inference
    run_inference_and_save_results(
        checkpoint_file=CHECKPOINT_FILE,
        image_paths=list_of_image_paths,
        output_base_dir=output_base_directory,
        device='cuda:0'
    )

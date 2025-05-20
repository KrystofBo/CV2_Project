import os
import os.path as osp
import shutil
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


def generate_masks(scene_folder, sam_checkpoint, model_type="vit_h", device="cuda", testing=False):
    train_output_folder = os.path.join("output", scene_folder, "train", "ours_30000")
    test_output_folder = os.path.join("output", scene_folder, "test", "ours_30000")

    gt_folder = os.path.join(train_output_folder, "gt")
    gt_test_folder = os.path.join(test_output_folder, "gt")

    id_folder = os.path.join(train_output_folder, "objects_id")
    id_test_folder = os.path.join(test_output_folder, "objects_id")

    refine_folder = create_refine_folder(scene_folder)

    output_folder = os.path.join("data", refine_folder, "object_mask")


    os.makedirs(output_folder, exist_ok=True)

    # Load SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Load image lists
    img_list = sorted([os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.png')])
    img_list += sorted([os.path.join(gt_test_folder, f) for f in os.listdir(gt_test_folder) if f.endswith('.png')])
    
    id_list = sorted([os.path.join(id_folder, f) for f in os.listdir(id_folder) if f.endswith('.png')])
    id_list += sorted([os.path.join(id_test_folder, f) for f in os.listdir(id_test_folder) if f.endswith('.png')])
    if testing:
        img_list = img_list[:3]
        id_list = id_list[:3]

    for img_idx, img_path in enumerate(tqdm(img_list, desc="Processing images")):
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # Load ID mask
        img_id = cv2.imread(id_list[img_idx], cv2.IMREAD_UNCHANGED)
        if img_id.ndim == 3:
            img_id = img_id[:, :, 0]  # Use red channel if RGB

        new_id_mask = np.ones_like(img_id, dtype=np.uint8) * 255
        unique_ids = np.unique(img_id)
        start_idx = 0
        obj_ids = unique_ids[start_idx:]

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

                masks, scores, _ = predictor.predict(
                    box=input_box,
                    multimask_output=False
                )

                if masks is not None and len(masks) > 0:
                    sam_mask = masks[0]
                    new_id_mask[sam_mask] = obj_id

        # Save new mask
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, new_id_mask)
        print(f"Saved new ID mask to: {output_path}")

    print("Finished creating new ID masks.")

def create_refine_folder(folder_name):
    """
    Creates a copy of the given folder with '_refine' suffix and clears the object_mask directory.
    
    Args:
        folder_name (str): Name of the data folder (e.g., 'office0', 'room0')
    
    Returns:
        str: Name of the created folder
    """
    source_folder = osp.join('data', folder_name)
    refine_folder_name = f"{folder_name}_refine"
    target_folder = osp.join('data', refine_folder_name)
    
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
    
    return refine_folder_name


def main():
    parser = argparse.ArgumentParser(description="Run SAM-based mask refinement using bounding boxes.")
    parser.add_argument("scene_folder", type=str, help="Path to the scene folder containing 'gt' and 'objects_id'")
    parser.add_argument("--checkpoint", type=str, default="sam_checkpoint/sam_vit_h_4b8939.pth",
                        help="Path to SAM model checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_h", help="Type of SAM model to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--test", action="store_true", help="Run only on 3 example images")

    args = parser.parse_args()
    generate_masks(args.scene_folder, args.checkpoint, args.model_type, args.device, args.test)


if __name__ == "__main__":
    main()

# example run:
# python mask_refiner_bb.py /home/tom.slik/data/cv2

# example run when testing:
# python mask_refiner_bb.py /home/tom.slik/data/cv2 --test

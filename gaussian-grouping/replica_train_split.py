#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path

def copy_middle_images(scene_name):
    # Define source and target directories
    base_dir = Path(__file__).parent
    scene_dir = base_dir / "data" / scene_name
    source_dir = scene_dir / "images"
    target_dir = scene_dir / "images_train"
    
    # Validate the scene directory exists
    if not os.path.exists(scene_dir):
        print(f"Error: Scene directory '{scene_dir}' does not exist.")
        return
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Validate source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    # Get all image files and sort them
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by number in filename
    
    # Calculate the start and end indices for middle 80%
    total_images = len(image_files)
    start_idx = int(total_images * 0.1)  # Skip first 10%
    end_idx = int(total_images * 0.9)    # Skip last 10%
    
    # Copy the middle 80% of images
    selected_images = image_files[start_idx:end_idx]
    
    print(f"Found {total_images} images in {scene_name}. Copying {len(selected_images)} images (the middle 80%)...")
    
    # Copy images to target directory
    for img_file in selected_images:
        source_path = source_dir / img_file
        target_path = target_dir / img_file
        shutil.copy2(str(source_path), str(target_path))
        
    print(f"Successfully copied {len(selected_images)} images to {target_dir}")

def main():
    parser = argparse.ArgumentParser(description='Copy middle 80% of images from a scene.')
    parser.add_argument('scene_name', type=str, help='Name of the scene (e.g., office, office1, room0)')
    args = parser.parse_args()
    
    copy_middle_images(args.scene_name)

if __name__ == "__main__":
    main() 
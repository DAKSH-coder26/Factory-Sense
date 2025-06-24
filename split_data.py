import os
import random
import shutil
from tqdm import tqdm

SOURCE_DIR = 'data'
TRAIN_RATIO = 0.8

def split_dataset():
    images_dir = os.path.join(SOURCE_DIR, 'images')
    labels_dir = os.path.join(SOURCE_DIR, 'labels')

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"Error: Source directories '{images_dir}' or '{labels_dir}' not found.")
        print("Please run the generate_data.py script first.")
        return

    train_img_dir = os.path.join(SOURCE_DIR, 'train', 'images')
    train_lbl_dir = os.path.join(SOURCE_DIR, 'train', 'labels')
    val_img_dir = os.path.join(SOURCE_DIR, 'val', 'images')
    val_lbl_dir = os.path.join(SOURCE_DIR, 'val', 'labels')

    for path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(path, exist_ok=True)

    filenames = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    random.shuffle(filenames)

    split_index = int(len(filenames) * TRAIN_RATIO)

    train_files = filenames[:split_index]
    val_files = filenames[split_index:]

    print(f"Total images: {len(filenames)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print("\nMoving training files...")
    for filename in tqdm(train_files):
        base_name = os.path.splitext(filename)[0]
        
        shutil.move(os.path.join(images_dir, filename), os.path.join(train_img_dir, filename))
        
        label_src_path = os.path.join(labels_dir, f"{base_name}.txt")
        if os.path.exists(label_src_path):
            shutil.move(label_src_path, os.path.join(train_lbl_dir, f"{base_name}.txt"))

    print("\nMoving validation files...")
    for filename in tqdm(val_files):
        base_name = os.path.splitext(filename)[0]
        shutil.move(os.path.join(images_dir, filename), os.path.join(val_img_dir, filename))

        label_src_path = os.path.join(labels_dir, f"{base_name}.txt")
        if os.path.exists(label_src_path):
            shutil.move(label_src_path, os.path.join(val_lbl_dir, f"{base_name}.txt"))


    print("\nDataset successfully split into training and validation sets.")
   
    try:
        os.rmdir(images_dir)
        os.rmdir(labels_dir)
    except OSError as e:
        print(f"Note: Could not remove original directories (they may not be empty): {e}")


if __name__ == '__main__':
    split_dataset()
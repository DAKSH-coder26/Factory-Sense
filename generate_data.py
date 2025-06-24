import cv2
import numpy as np
import time
import os
import random
from simulation_elements import (
    VirtualBottle,
    draw_environment, 
    BACKGROUND_PATHS,          
    WIDTH,
    HEIGHT
)

# --- Configuration ---
MAX_FRAMES = 5000  
SPAWN_INTERVAL = 5 

CLASS_MAPPING = {
    'bottle_empty': 0,
    'bottle_filling': 1,
    'bottle_filled': 2,
    'bottle_capped': 3,
    'bottle_labeled': 4
}

def setup_directories():
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/labels', exist_ok=True)

def main():
    setup_directories()

    bottles_on_belt = []
    next_bottle_id = 1
    last_spawn_time = time.time()
    last_frame_time = time.time()
    frame_count = 0

    print(f"Starting data generation for {MAX_FRAMES} frames...")

    while frame_count < MAX_FRAMES:
        current_time = time.time()
        delta_time = current_time - last_frame_time
        if delta_time == 0: continue
        last_frame_time = current_time

        if BACKGROUND_PATHS:
           
            bg_path = random.choice(BACKGROUND_PATHS)
            frame = cv2.imread(bg_path)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
        else:
            
            frame = np.full((HEIGHT, WIDTH, 3), (60, 60, 60), dtype=np.uint8)

        
        draw_environment(frame)

        if current_time - last_spawn_time > SPAWN_INTERVAL:
            
            bottles_on_belt.append(VirtualBottle(next_bottle_id))
            next_bottle_id += 1
            last_spawn_time = current_time

        label_lines = []
        for bottle in bottles_on_belt:
            bottle.update_position(delta_time)
            bottle.update_state()
            bottle.draw(frame)

            data = bottle.get_tracker_format()
            bbox = data['bbox']
            
            if data['label'] in CLASS_MAPPING:
                class_id = CLASS_MAPPING[data['label']]
                
                x1, y1, x2, y2 = bbox
                box_width = x2 - x1
                box_height = y2 - y1
                x_center = x1 + box_width / 2
                y_center = y1 + box_height / 2
                
                x_center_norm = x_center / WIDTH
                y_center_norm = y_center / HEIGHT
                width_norm = box_width / WIDTH
                height_norm = box_height / HEIGHT
                
                label_lines.append(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

        
        if random.random() < 0.5: 
            frame = cv2.GaussianBlur(frame, (5,5), 0)

        image_path = f'data/images/frame_{frame_count:05d}.png'
        label_path = f'data/labels/frame_{frame_count:05d}.txt'

        cv2.imwrite(image_path, frame)
        if label_lines: 
            with open(label_path, 'w') as f:
                f.writelines(label_lines)

        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Generated {frame_count}/{MAX_FRAMES} frames...")

    print(f"\nData generation complete. {MAX_FRAMES} images and labels saved in the 'data/' directory.")

if __name__ == '__main__':
    main()
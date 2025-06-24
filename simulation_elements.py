import cv2
import numpy as np
import random
import os
import time

WIDTH, HEIGHT = 1280, 500
CONVEYOR_Y = 420
BELT_SPEED = 60

ZONES = {
    'filling': {'x_range': (250, 450), 'color': (255, 220, 220, 0.5)},
    'capping': {'x_range': (550, 750), 'color': (220, 255, 220, 0.5)},
    'labeling': {'x_range': (850, 1050), 'color': (220, 220, 255, 0.5)}
}

try:
    BACKGROUND_PATHS = [os.path.join('backgrounds', f) for f in os.listdir('backgrounds') if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not BACKGROUND_PATHS:
        print("Warning: 'backgrounds' folder is empty. Using plain background.")
    else:
        print(f"Found {len(BACKGROUND_PATHS)} background images.")
except FileNotFoundError:
    print("Warning: 'backgrounds' folder not found. Using plain background.")
    BACKGROUND_PATHS = []


def overlay_transparent_image(background, overlay, x, y):
    h, w = overlay.shape[0], overlay.shape[1]
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)
    overlay_y1, overlay_y2 = max(0, -y), min(h, background.shape[0] - y)
    overlay_x1, overlay_x2 = max(0, -x), min(w, background.shape[1] - x)
    roi = background[y1:y2, x1:x2]
    roi_overlay = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    if roi.shape[0] == 0 or roi.shape[1] == 0 or roi_overlay.shape[0] == 0 or roi_overlay.shape[1] == 0:
        return
    overlay_bgr = roi_overlay[:,:,0:3]
    alpha_mask = roi_overlay[:,:,3] / 255.0
    inverse_alpha_mask = 1.0 - alpha_mask
    roi_bg = (roi * np.expand_dims(inverse_alpha_mask, axis=2)).astype(np.uint8)
    roi_fg = (overlay_bgr * np.expand_dims(alpha_mask, axis=2)).astype(np.uint8)
    background[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)


class VirtualBottle:
    try:
        BOTTLE_IMAGE = cv2.imread('bottle1.png', cv2.IMREAD_UNCHANGED)
    except (FileNotFoundError, IndexError):
        BOTTLE_IMAGE = None

    def __init__(self, bottle_id, anomaly_type=None):
        self.id = bottle_id
        self.x = 0
        self.y = CONVEYOR_Y
        self.state = 'bottle_empty'
        self.image_to_draw = None
        
        self.anomaly_type = anomaly_type
        self.stuck_info = {'is_stuck': False, 'stuck_until': 0}

        if self.BOTTLE_IMAGE is not None:
            scale = 0.5  
            original_h, original_w = self.BOTTLE_IMAGE.shape[:2]
            self.height = int(original_h * scale)
            self.width = int(original_w * scale)
            self.image_to_draw = cv2.resize(self.BOTTLE_IMAGE, (self.width, self.height))
        else:
            self.width = 30
            self.height = 50 
            self.color = (180, 105, 50)

    def update_position(self, delta_time):
        if self.anomaly_type == 'stuck':
            center_x = self.x + self.width // 2
            if ZONES['filling']['x_range'][0] < center_x < ZONES['filling']['x_range'][1] and not self.stuck_info['is_stuck']:
                self.stuck_info['is_stuck'] = True
                self.stuck_info['stuck_until'] = time.time() + 2
                print(f"[ANOMALY SCRIPT] Bottle {self.id} is now stuck in the filling zone.")

        if self.stuck_info['is_stuck'] and time.time() < self.stuck_info['stuck_until']:
            return

        current_speed = BELT_SPEED
        if self.anomaly_type == 'misaligned':
            center_x = self.x + self.width // 2
            if ZONES['capping']['x_range'][1] < center_x < ZONES['labeling']['x_range'][0]:
                current_speed *= 2.5

        self.x += int(current_speed * delta_time)

    def update_state(self):
        center_x = self.x + self.width // 2

        if self.anomaly_type == 'missing_label':
            if ZONES['labeling']['x_range'][0] <= center_x < ZONES['labeling']['x_range'][1]:
                self.state = 'bottle_capped'
                return
        if center_x < ZONES['filling']['x_range'][0]:
            self.state = 'bottle_empty'
        elif center_x < ZONES['capping']['x_range'][0]:
            self.state = 'bottle_filled'
        elif center_x < ZONES['labeling']['x_range'][0]:
            self.state = 'bottle_capped'
        else:
            self.state = 'bottle_labeled'

        if ZONES['filling']['x_range'][0] <= center_x < ZONES['filling']['x_range'][1]:
            self.state = 'bottle_filling'
        elif ZONES['capping']['x_range'][0] <= center_x < ZONES['capping']['x_range'][1]:
            self.state = 'bottle_capped'
        elif ZONES['labeling']['x_range'][0] <= center_x < ZONES['labeling']['x_range'][1]:
            self.state = 'bottle_labeled'

    def get_tracker_format(self):
        x1, y1 = self.x, self.y - self.height
        x2, y2 = self.x + self.width, self.y
        return {'id': self.id, 'label': self.state, 'bbox': [x1, y1, x2, y2], 'conf': 0.99}

    def draw(self, frame):
        if self.image_to_draw is not None:
            draw_y = self.y - self.height
            overlay_transparent_image(frame, self.image_to_draw, self.x, draw_y)
        else:
            x1, y1, x2, y2 = self.get_tracker_format()['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 1)

def draw_environment(frame):
    overlay = frame.copy()
    for zone_name, props in ZONES.items():
        x1, x2 = props['x_range']
        color = props['color'][:3]
        cv2.rectangle(overlay, (x1, 50), (x2, HEIGHT - 100), color, -1)
        cv2.putText(frame, zone_name.upper(), (x1 + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.line(frame, (0, CONVEYOR_Y), (WIDTH, CONVEYOR_Y), (0, 0, 0), 2)
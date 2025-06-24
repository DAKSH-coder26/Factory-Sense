import cv2
import numpy as np
import time
import random
import os
from yolo_detector import YoloDetector
from deep_sort_tracker import DeepSortTracker
from bottle_tracker import BottleTracker
from anomaly_detector import AnomalyDetector
from llm_reasoner import LLMReasoner
from scenario_generator import ScenarioGenerator 

from simulation_elements import (
    draw_environment,
    BACKGROUND_PATHS,
    WIDTH,
    HEIGHT
)

MODEL_PATH = 'best.pt'
CONFIDENCE_THRESHOLD = 0.7
ZONES = {
    'filling': (250, 450),
    'capping': (550, 750),
    'labeling': (850, 1050)
}
TARGET_FPS = 30
FRAME_DURATION = 1.0 / TARGET_FPS

def main():
    print("Initializing system components...")
    cv2.namedWindow('FactorySense - Live Simulation', cv2.WINDOW_NORMAL)

    if BACKGROUND_PATHS:
        selected_bg_path = random.choice(BACKGROUND_PATHS)
        static_background = cv2.imread(selected_bg_path)
        static_background = cv2.resize(static_background, (WIDTH, HEIGHT))
    else:
        static_background = np.full((HEIGHT, WIDTH, 3), (60, 60, 60), dtype=np.uint8)

    loading_frame = static_background.copy()
    loading_text = "Initializing... Please Wait."
    text_size, _ = cv2.getTextSize(loading_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    text_x = (WIDTH - text_size[0]) // 2
    text_y = (HEIGHT + text_size[1]) // 2
    cv2.putText(loading_frame, loading_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.imshow('FactorySense - Live Simulation', loading_frame)
    cv2.waitKey(1)
    
    detector = YoloDetector(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
    tracker = DeepSortTracker()
    bottle_tracker = BottleTracker(max_history=50)
    anomaly_detector = AnomalyDetector(zones=ZONES)
    llm_reasoner = LLMReasoner()

    scenario = ScenarioGenerator(total_bottles=5, spawn_interval=10)

    reported_anomalies = set()
    on_screen_alerts = {}

    print("System initialized. Starting simulation...")
    while True:
        loop_start_time = time.time()
        frame = static_background.copy()
        draw_environment(frame)

        bottles_on_belt = scenario.update()

        for bottle in bottles_on_belt:
            bottle.update_position(FRAME_DURATION)
            bottle.update_state()
            bottle.draw(frame)
        
        bottles_on_belt[:] = [b for b in bottles_on_belt if b.x < WIDTH]
        
        if scenario.is_complete():
            print("Scenario complete. All bottles have run. Ending simulation.")
            break
        
        ground_truth_states = {b.id: b.state for b in bottles_on_belt}
        detections = detector.detect(frame)
        tracked_objects = tracker.update_tracks(detections, frame=frame)
        bottle_tracker.update(tracked_objects)

        current_anomalies = []
        for bottle_id in bottle_tracker.get_all_bottles():
            history = bottle_tracker.get_state_history(bottle_id)
            position_history = bottle_tracker.get_position_history(bottle_id)
            if history and position_history:
                anomalies = anomaly_detector.check_anomalies(bottle_id, history, position_history)
                if anomalies:
                    current_anomalies.extend(anomalies)
        
        anomalous_ids = {a['bottle_id'] for a in current_anomalies}

        for anomaly in current_anomalies:
            anomaly_key = (anomaly['bottle_id'], anomaly['type'])
            if anomaly_key not in reported_anomalies:
                print(f"[ALERT] New anomaly detected: {anomaly}")
                explanation = llm_reasoner.explain_anomalies([anomaly])[0]
                on_screen_alerts[anomaly['bottle_id']] = explanation
                reported_anomalies.add(anomaly_key)
        
        active_ids = {obj['id'] for obj in tracked_objects}
        for bottle_id in list(on_screen_alerts.keys()):
            if bottle_id not in active_ids or bottle_id not in anomalous_ids:
                del on_screen_alerts[bottle_id]

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            track_id = obj['id']
            detected_label = obj['label']

            box_color = (0, 255, 0) 
            if track_id in anomalous_ids:
                box_color = (0, 0, 255) 

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            gt_state = ground_truth_states.get(track_id, "N/A")
            display_text = f"ID: {track_id} [Detected: {detected_label} | GT: {gt_state}]"
            
            text_color = (255, 255, 255)
            if detected_label != gt_state:
                text_color = (0, 255, 255) 

            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        alert_y_pos = 30
        for bottle_id, text in on_screen_alerts.items():
            alert_text = f"ALERT [ID: {bottle_id}]: {text}"
            cv2.putText(frame, alert_text, (10, alert_y_pos), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 220), 2)
            alert_y_pos += 25

        cv2.imshow('FactorySense - Live Simulation', frame)
        
        time_elapsed = time.time() - loop_start_time
        sleep_time = FRAME_DURATION - time_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Processing finished. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
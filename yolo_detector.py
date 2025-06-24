import torch
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"YOLO Detector using device: {self.device}")
        
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'conf': score,
                'label': self.model.names[int(class_id)]
            })
            
        return detections
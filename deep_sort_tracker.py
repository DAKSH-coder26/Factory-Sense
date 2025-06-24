from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class DeepSortTracker:
    def __init__(self, max_age=60, n_init=3, max_cosine_distance=0.6):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_cosine_distance=max_cosine_distance)

    def update_tracks(self, detections, frame=None):
        formatted_detections = []
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            generic_class_for_tracker = 'bottle'
            formatted_detections.append(
                ([int(x1), int(y1), int(width), int(height)], det['conf'], generic_class_for_tracker)
            )

        tracked_objects = self.tracker.update_tracks(formatted_detections, frame=frame)

        output = []
        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            try:
                track_id = track.track_id
                l, t, w, h = track.to_ltwh()

                original_detection = self._find_best_match_by_iou([l,t,w,h], detections)
                if original_detection:
                    specific_label = original_detection['label']
                    output.append({
                        'id': track_id,
                        'label': specific_label,
                        'bbox': [int(l), int(t), int(l + w), int(t + h)],
                        'conf': round(track.det_conf, 2)
                    })
            except Exception as e:
                print(f"[WARN] Could not process track: {e}")

        return output

    def _calculate_iou(self, boxA, boxB):
        boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _find_best_match_by_iou(self, track_bbox, detections):
        best_iou = 0.0
        best_match = None
        iou_threshold = 0.3 

        print(f"[TRACKER_DEBUG] Finding match for track_bbox: {[int(x) for x in track_bbox]}")

        for det in detections:
            iou = self._calculate_iou(track_bbox, det['bbox'])

            print(f"[TRACKER_DEBUG]  ... vs detection_bbox: {det['bbox']} (label: {det['label']}) -> IoU: {iou:.2f}")

            if iou > best_iou:
                best_iou = iou
                best_match = det

        if best_iou > iou_threshold:
            
            print(f"[TRACKER_SUCCESS] Match found with IoU: {best_iou:.2f}! Assigning label: '{best_match['label']}'")
            return best_match
        if detections: 
            print(f"[TRACKER_FAILURE] No suitable match found for track. Best IoU was {best_iou:.2f}, but threshold is {iou_threshold}.")

        return None
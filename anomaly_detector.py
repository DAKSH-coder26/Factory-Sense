EXPECTED_SEQUENCE = [
    'bottle_empty',
    'bottle_filling',
    'bottle_filled',
    'bottle_capped',
    'bottle_labeled'
]

class AnomalyDetector:
    def __init__(self, zones=None):
        self.zones = zones or {}

    def check_anomalies(self, bottle_id, state_history, position_history):
        anomalies = []
        if not position_history: 
            return []
            
        current_position = position_history[-1]

        index_list = [EXPECTED_SEQUENCE.index(s) for s in state_history if s in EXPECTED_SEQUENCE]
        if index_list != sorted(index_list):
            anomalies.append({
                "bottle_id": bottle_id,
                "type": "Stage out of order",
                "details": state_history
            })

        if self.zones and 'bottle_capped' in state_history and 'bottle_labeled' not in state_history:
             labeling_zone_end = self.zones.get('labeling', (0, 0))[1]
             if current_position[0] > labeling_zone_end + 50:
                anomalies.append({
                    "bottle_id": bottle_id,
                    "type": "Label missing",
                    "details": state_history
                })

        STUCK_FRAME_COUNT = 10
        STUCK_PIXEL_THRESHOLD = 5 
        if len(state_history) > STUCK_FRAME_COUNT:
            last_n_states = state_history[-STUCK_FRAME_COUNT:]
           
            if len(set(last_n_states)) == 1:
                last_n_positions = position_history[-STUCK_FRAME_COUNT:]
                
                start_x = last_n_positions[0][0]
                end_x = last_n_positions[-1][0]
                distance_moved = abs(end_x - start_x)

                if distance_moved < STUCK_PIXEL_THRESHOLD:
                    stuck_state = last_n_states[0]
                    anomalies.append({
                        "bottle_id": bottle_id,
                        "type": "Stuck bottle",
                        "details": f"Stuck in state '{stuck_state}' and position for {STUCK_FRAME_COUNT} frames."
                    })

        if self.zones and state_history:
            last_state = state_history[-1]
            zone = self._zone_for_state(last_state)
            if zone and current_position:
                x = current_position[0]
                zone_x1, zone_x2 = self.zones.get(zone, (None, None))
                if zone_x1 is not None and not (zone_x1 <= x <= zone_x2):
                    anomalies.append({
                        "bottle_id": bottle_id,
                        "type": "Misaligned in " + zone,
                        "position": x,
                        "expected_range": (zone_x1, zone_x2)
                    })

        return anomalies

    def _zone_for_state(self, state):
        if 'filling' in state: return 'filling'
        if 'capped' in state: return 'capping'
        if 'labeled' in state: return 'labeling'
        return None
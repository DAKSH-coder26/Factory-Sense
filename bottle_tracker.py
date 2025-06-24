from collections import defaultdict, deque

class BottleTracker:
    def __init__(self, max_history=30):
        self.bottle_states = defaultdict(lambda: deque(maxlen=max_history))
        self.bottle_positions = {}  
        self.bottle_position_history = defaultdict(lambda: deque(maxlen=max_history)) 

    def update(self, tracked_bottles):
        
        for bottle in tracked_bottles:
            bottle_id = bottle['id']
            label = bottle['label']
            bbox = bottle['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            current_position = (center_x, center_y)

            self.bottle_states[bottle_id].append(label)
            self.bottle_positions[bottle_id] = current_position
            self.bottle_position_history[bottle_id].append(current_position) 

    def get_state_history(self, bottle_id):
        return list(self.bottle_states[bottle_id])

    def get_position(self, bottle_id):
        return self.bottle_positions.get(bottle_id, None)

    def get_position_history(self, bottle_id):
        return list(self.bottle_position_history[bottle_id])

    def get_all_bottles(self):
        return list(self.bottle_states.keys())
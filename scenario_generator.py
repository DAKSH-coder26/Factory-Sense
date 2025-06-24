# scenario_generator.py

import time
import random
from simulation_elements import VirtualBottle

class ScenarioGenerator:
    def __init__(self, total_bottles=5, spawn_interval=10):
        
        self.total_bottles_to_spawn = total_bottles
        self.spawn_interval = spawn_interval
        
        self.bottles_on_belt = []
        self.bottles_spawned_count = 0
        self.last_spawn_time = time.time() - self.spawn_interval
        
        self.anomaly_types = ['stuck', 'misaligned', 'missing_label']
        self.bottle_scripts = [None] * (total_bottles - 3) + self.anomaly_types
        random.shuffle(self.bottle_scripts)
        print(f"[SCENARIO] Initialized. Bottle anomaly plan: {self.bottle_scripts}")

    def update(self):

        if self.bottles_spawned_count < self.total_bottles_to_spawn and \
           (time.time() - self.last_spawn_time) > self.spawn_interval:
            
            self.bottles_spawned_count += 1
            anomaly_type_for_this_bottle = self.bottle_scripts.pop()

            if anomaly_type_for_this_bottle:
                print(f"[SCENARIO] Spawning bottle {self.bottles_spawned_count} with SCRIPTED ANOMALY: {anomaly_type_for_this_bottle}.")
            else:
                print(f"[SCENARIO] Spawning bottle {self.bottles_spawned_count} (Normal).")

            new_bottle = VirtualBottle(self.bottles_spawned_count, anomaly_type=anomaly_type_for_this_bottle)
            self.bottles_on_belt.append(new_bottle)
            self.last_spawn_time = time.time()
            
        return self.bottles_on_belt

    def is_complete(self):
        return self.bottles_spawned_count == self.total_bottles_to_spawn and not self.bottles_on_belt
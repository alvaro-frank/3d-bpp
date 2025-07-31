import random

class RandomAgent:
    def __init__(self, bin_width, bin_height, bin_depth):
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.bin_depth = bin_depth

    def get_action(self, box):
        rotation = random.randint(0, 5)
        w, h, d = box.rotate(rotation)
        
        x = random.randint(0, max(0, self.bin_width - w))
        y = random.randint(0, max(0, self.bin_height - h))

        return [x, y, rotation]

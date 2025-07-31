import numpy as np

class Bin:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.boxes = []

    def bin_volume(self):
        w, h, d = self.bin_size
        return w * h * d

    def fits(self, box_dims, position):
        x, y, z = position
        bw, bh, bd = box_dims
        return (x + bw <= self.width and
                y + bh <= self.height and
                z + bd <= self.depth)

    def collides(self, box_dims, position):
        x1, y1, z1 = position
        bw1, bh1, bd1 = box_dims

        for b in self.boxes:
            x2, y2, z2 = b.position
            bw2, bh2, bd2 = b.rotate(b.rotation_type)

            if (x1 < x2 + bw2 and x1 + bw1 > x2 and
                y1 < y2 + bh2 and y1 + bh1 > y2 and
                z1 < z2 + bd2 and z1 + bd1 > z2):
                return True
        return False

    def place_box(self, box, position, rotation_type):
        x, y = position
        bw, bh, bd = box.rotate(rotation_type)
        z = self.find_lowest_z((bw, bh, bd), x, y)

        # Verifica se cabe dentro do bin
        if x + bw > self.width or y + bh > self.height or z + bd > self.depth:
            return False

        # Verifica colisão com outras caixas
        if self.collides((bw, bh, bd), (x, y, z)):
            return False

        # Posiciona a caixa
        box.rotation_type = rotation_type
        box.place_at(x, y, z)
        self.boxes.append(box)
        return True

    def get_placed_boxes_volume(self):
        volume = 0
        for box in self.boxes:  # ou qualquer estrutura que guarde as caixas colocadas
            volume += box.get_volume()
        return volume

    def find_lowest_z(self, box_dims, x, y):
        max_z = 0
        bw, bh, bd = box_dims

        for b in self.boxes:
            bx, by, bz = b.position
            bw2, bh2, bd2 = b.rotate(b.rotation_type)

            overlap_x = not (x + bw <= bx or x >= bx + bw2)
            overlap_y = not (y + bh <= by or y >= by + bh2)

            if overlap_x and overlap_y:
                top_z = bz + bd2
                if top_z > max_z:
                    max_z = top_z

        return max_z

    def calculate_compactness(self, placed_box):
        if len(self.boxes) <= 1:
            return 0.0  # first box, no neighbors to cluster with

        x1, y1, z1 = placed_box.position
        distances = []
        for box in self.boxes:
            if box == placed_box:
                continue
            x2, y2, z2 = box.position
            dist = np.linalg.norm([x1 - x2, y1 - y2, z1 - z2])
            distances.append(dist)

        if distances:
            avg_dist = sum(distances) / len(distances)
            return max(0.0, 1.0 - avg_dist / max(self.width, self.height, self.depth))
        else:
            return 0.0
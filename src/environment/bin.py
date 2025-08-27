import numpy as np

"""
Represents a 3D bin where boxes can be placed.
It has a width, height, depth, and can hold multiple boxes.
The bin can check if a box fits and if it collides with other boxes.
"""
class Bin:
    def __init__(self, width, depth, height):
        """
        Initializes a bin with given width (X), depth (Y), height (Z).
        """
        self.width = width   # X-axis
        self.depth = depth   # Y-axis
        self.height = height # Z-axis
        self.boxes = []

    def bin_volume(self):
        """Returns the total volume of the bin."""
        return self.width * self.depth * self.height

    def fits(self, box_dims, position):
        """
        Checks if a box fits inside the bin boundaries.
        
        Parameters:
        - box_dims (tuple): (width, depth, height)
        - position (tuple): (x, y, z) inside the bin

        Returns:
        - bool: True if the box fits entirely, False otherwise
        """
        x, y, z = position
        bw, bd, bh = box_dims  # width, depth, height
        return (x + bw <= self.width and
                y + bd <= self.depth and
                z + bh <= self.height)

    def collides(self, box_dims, position):
        """
        Checks if a box collides with any existing boxes.
        
        Parameters:
        - box_dims (tuple): (width, depth, height)
        - position (tuple): (x, y, z)

        Returns:
        - bool: True if collision occurs, False otherwise
        """
        x1, y1, z1 = position
        bw1, bd1, bh1 = box_dims

        for b in self.boxes:
            x2, y2, z2 = b.position
            bw2, bd2, bh2 = b.rotate(b.rotation_type)

            overlap_x = (x1 < x2 + bw2) and (x1 + bw1 > x2)
            overlap_y = (y1 < y2 + bd2) and (y1 + bd1 > y2)
            overlap_z = (z1 < z2 + bh2) and (z1 + bh1 > z2)

            if overlap_x and overlap_y and overlap_z:
                return True
        return False

    def place_box(self, box, position, rotation_type):
        """
        Attempt to place a box at (x,y) with lowest z possible.
        
        Parameters:
        - box (Box)
        - position (tuple): (x, y) base position
        - rotation_type (int): rotation index

        Returns:
        - bool: True if placed successfully, False otherwise
        """
        x, y = position
        bw, bd, bh = box.rotate(rotation_type)  # rotated dims
        z = self.find_lowest_z((bw, bd, bh), x, y)

        # Check bin bounds
        if x + bw > self.width or y + bd > self.depth or z + bh > self.height:
            return False

        # Check collisions
        if self.collides((bw, bd, bh), (x, y, z)):
            return False

        # Place the box
        box.rotation_type = rotation_type
        box.place_at(x, y, z)
        self.boxes.append(box)
        return True

    def find_lowest_z(self, box_dims, x, y):
        """
        Finds the lowest Z (height) where a box can be placed at (x,y).
        It checks stacking over other boxes.
        
        Parameters:
        - box_dims (tuple): (width, depth, height)
        - x (int), y (int): X and Y base positions

        Returns:
        - int: lowest Z position where the box can be placed
        """
        max_z = 0
        bw, bd, bh = box_dims

        for b in self.boxes:
            bx, by, bz = b.position
            bw2, bd2, bh2 = b.rotate(b.rotation_type)

            overlap_x = not (x + bw <= bx or x >= bx + bw2)
            overlap_y = not (y + bd <= by or y >= by + bd2)

            if overlap_x and overlap_y:
                top_z = bz + bh2
                if top_z > max_z:
                    max_z = top_z

        return max_z
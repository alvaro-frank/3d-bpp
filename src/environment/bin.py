import numpy as np

"""
Represents a 3D bin where boxes can be placed.
It has a width, height, depth, and can hold multiple boxes.
The bin can check if a box fits and if it collides with other boxes.
"""
class Bin:
    def __init__(self, width, height, depth):
        """
        Initializes a bin with given width, height, depth.
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.boxes = []

    def bin_volume(self):
        """
        Calculates the volume of the bin (width × height × depth).
        
        Returns:
        - int: the total volume of the bin
        """
        return self.width * self.height * self.depth

    def fits(self, box_dims, position):
        """
        Checks if a box can fit in the bin at a specific position.
        
        Parameters:
        - box_dims (tuple): dimensions of the box (width, height, depth)
        - position (tuple): position in the bin (x, y, z)
        
        Returns:
        - bool: True if the box fits, False otherwise
        """
        x, y, z = position
        bw, bh, bd = box_dims
        return (x + bw <= self.width and # fits in X-axis
                y + bh <= self.height and # fits in Y-axis
                z + bd <= self.depth) # fits in Z-axis

    def collides(self, box_dims, position):
        """
        Checks if a box collides with any existing boxes in the bin.
        
        Parameters:
        - box_dims (tuple): dimensions of the box (width, height, depth)
        - position (tuple): position in the bin (x, y, z)
        
        Returns:
        - bool: True if there is a collision, False otherwise
        """
        x1, y1, z1 = position
        bw1, bh1, bd1 = box_dims

        for b in self.boxes: # for each box in the bin
            x2, y2, z2 = b.position # get its position
            bw2, bh2, bd2 = b.rotate(b.rotation_type) # get its dimensions after rotation

            if (x1 < x2 + bw2 and x1 + bw1 > x2 and # check X-axis overlap
                y1 < y2 + bh2 and y1 + bh1 > y2 and # check Y-axis overlap
                z1 < z2 + bd2 and z1 + bd1 > z2): # check Z-axis overlap
                return True
        return False

    def place_box(self, box, position, rotation_type):
        """
        Places a box in the bin at a specific position and rotation.
        
        Parameters:
        - box (Box): the box to place
        - position (tuple): position in the bin (x, y)
        - rotation_type (int): rotation index [0–5]
        
        Returns:
        - bool: True if the box was placed successfully, False otherwise
        """
        x, y = position
        bw, bh, bd = box.rotate(rotation_type) # get rotated dimensions
        z = self.find_lowest_z((bw, bh, bd), x, y) # find the lowest Z position

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

    def find_lowest_z(self, box_dims, x, y):
        """
        Finds the lowest Z position where a box can be placed at (x, y).
        It checks the top of existing boxes at that (x, y) position.
        
        Parameters:
        - box_dims (tuple): dimensions of the box (width, height, depth)
        - x (int): X position in the bin
        - y (int): Y position in the bin
        
        Returns:
        - int: the lowest Z position where the box can be placed
        """
        max_z = 0
        bw, bh, bd = box_dims

        for b in self.boxes:
            bx, by, bz = b.position
            bw2, bh2, bd2 = b.rotate(b.rotation_type)

            overlap_x = not (x + bw <= bx or x >= bx + bw2) # check X-axis overlap
            overlap_y = not (y + bh <= by or y >= by + bh2) # check Y-axis overlap

            if overlap_x and overlap_y:
                top_z = bz + bd2
                if top_z > max_z:
                    max_z = top_z

        return max_z
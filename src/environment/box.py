"""
Represents a box to be placed in a bin.
It has a width, height, depth, and an optional ID.
The box can be rotated in 6 different ways.
The position in the bin is stored as a tuple (x, y, z).
"""
class Box:
    def __init__(self, width, height, depth, id=None):
        """
        Initializes a box with given width, height, depth
    
        Parameters:
        - width (int): size of the box along the X-axis
        - height (int): size of the box along the Y-axis
        - depth (int): size of the box along the Z-axis
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.position = None
        self.rotation_type = 0
        self.id = id

    def get_volume(self):
        """
        Calculates the volume of the box (width × height × depth).

        Returns:
        - int: the total volume of the box
        """
        return self.width * self.height * self.depth

    def place_at(self, x, y, z):
        """
        Place the box at a specific position inside the bin.

        Parameters:
        - x (int): position along the bin's X-axis
        - y (int): position along the bin's Y-axis
        - z (int): position along the bin's Z-axis
        """
        self.position = (x, y, z)

    def rotate(self, rotation_type):
        """
        Rotate the box into one of 6 possible orientations.
        Each orientation swaps the dimensions (width, height, depth).

        Parameters:
        - rotation_type (int): rotation index [0–5]

        Returns:
        - tuple: rotated (width, height, depth)
        """
        w, h, d = self.width, self.height, self.depth
        
        if rotation_type == 0: # No rotation
            return (w, h, d)
        elif rotation_type == 1: # Rotate 90 degrees around X-axis
            return (w, d, h)
        elif rotation_type == 2: # Rotate 90 degrees around Y-axis
            return (h, w, d)
        elif rotation_type == 3: # Rotate 90 degrees around Z-axis
            return (h, d, w)
        elif rotation_type == 4: # Rotate 180 degrees around X-axis
            return (d, w, h)
        elif rotation_type == 5: # Rotate 180 degrees around Y-axis
            return (d, h, w)

    def get_rotated_size(self):
        """
        Get the box dimensions according to its current rotation type.

        Returns:
        - tuple: (width, height, depth) after applying the current rotation
        """
        return self.rotate(self.rotation_type)

class Box:
    """
    Represents a box to be placed in a bin.
    It has a width (X), depth (Y), and height (Z), and an optional ID.
    The box can be rotated in 6 different orientations.
    The position in the bin is stored as a tuple (x, y, z).
    """
    def __init__(self, width, depth, height, id=None):
        """
        Initializes a box with given width, depth, height.

        Args:
        - width (int): size of the box along the X-axis
        - depth (int): size of the box along the Y-axis
        - height (int): size of the box along the Z-axis
        """
        self.width = width   # X-axis
        self.depth = depth   # Y-axis
        self.height = height # Z-axis
        self.position = None
        self.rotation_type = 0
        self.id = id

    def get_volume(self):
        """Returns the volume of the box."""
        return self.width * self.depth * self.height

    def place_at(self, x, y, z):
        """
        Place the box at a specific position inside the bin.

        Args:
        - x (int): position along the bin's X-axis
        - y (int): position along the bin's Y-axis
        - z (int): position along the bin's Z-axis
        """
        self.position = (x, y, z)

    def rotate(self, rotation_type):
        """
        Rotate the box into one of 6 possible orientations.
        Each orientation swaps (width=X, depth=Y, height=Z).

        Args:
        - rotation_type (int): rotation index [0–5]

        Returns:
        - tuple: rotated (width, depth, height)
        """
        w, d, h = self.width, self.depth, self.height

        if rotation_type == 0:   # No rotation
            return (w, d, h)
        elif rotation_type == 1: # Rotate swapping depth and height
            return (w, h, d)
        elif rotation_type == 2: # Rotate swapping width and depth
            return (d, w, h)
        elif rotation_type == 3: # Rotate swapping width and height
            return (h, d, w)
        elif rotation_type == 4: # Rotate (depth, height, width)
            return (d, h, w)
        elif rotation_type == 5: # Rotate (height, width, depth)
            return (h, w, d)

    def get_rotated_size(self):
        """
        Get the box dimensions according to its current rotation type.
        """
        return self.rotate(self.rotation_type)

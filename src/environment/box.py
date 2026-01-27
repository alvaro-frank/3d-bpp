# ==============================================================================
# FILE: environment/box.py
# DESCRIPTION: Defines the Box class for 3D Bin Packing.
#              Supports 6-DOF rotations and spatial state management.
# ==============================================================================

class Box:
    """
    Represents an item to be placed within a 3D Bin.
    Tracks physical dimensions, orientation, and spatial position (x, y, z).
    """
    def __init__(self, width, depth, height, id=None):
        """
        Initializes a box with its base dimensions.

        Args:
            width (int/float): Initial size along the X-axis.
            depth (int/float): Initial size along the Y-axis.
            height (int/float): Initial size along the Z-axis.
            id (Optional): Unique identifier for the box.
        """
        self.width = width   # X-axis
        self.depth = depth   # Y-axis
        self.height = height # Z-axis
        self.position = None
        self.rotation_type = 0
        self.id = id

    def get_volume(self):
        """Returns the total volume of the box."""
        return self.width * self.depth * self.height

    # --------------------------------------------------------------------------
    # SPATIAL & STATE MANAGEMENT
    # --------------------------------------------------------------------------

    def place_at(self, x, y, z):
        """
        Update the box's spatial coordinates within a bin.

        Args:
            x (int): Final position along the bin's X-axis.
            y (int): Final position along the bin's Y-axis.
            z (int): Final position along the bin's Z-axis.
        """
        self.position = (x, y, z)

    def get_rotated_size(self):
        """
        Get the current dimensions of the box based on its rotation_type.
        
        Returns:
            tuple: (width, depth, height) after applying current rotation.
        """
        return self.rotate(self.rotation_type)

    # --------------------------------------------------------------------------
    # ROTATION LOGIC
    # --------------------------------------------------------------------------

    def rotate(self, rotation_type):
        """
        Calculates the dimensions for one of the 6 possible 3D orientations.
        Swaps width (X), depth (Y), and height (Z) axes.

        Args:
            rotation_type (int): Orientation index from 0 to 5.

        Returns:
            tuple: Rotated (width, depth, height).
        """
        w, d, h = self.width, self.depth, self.height

        # Each case represents a unique permutation of (W, D, H)
        if rotation_type == 0:   # (W, D, H) - Default
            return (w, d, h)
        elif rotation_type == 1: # (W, H, D)
            return (w, h, d)
        elif rotation_type == 2: # (D, W, H)
            return (d, w, h)
        elif rotation_type == 3: # (H, D, W)
            return (h, d, w)
        elif rotation_type == 4: # (D, H, W)
            return (d, h, w)
        elif rotation_type == 5: # (H, W, D)
            return (h, w, d)
        
        return (w, d, h)
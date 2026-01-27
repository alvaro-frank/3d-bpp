# ==============================================================================
# FILE: utils/action_space.py
# DESCRIPTION: Logic for generating the discrete action space of the bin.
#              Maps grid coordinates and rotations to a flat action index.
# ==============================================================================

def generate_discrete_actions(bin_width, bin_depth, rotations=[0, 1, 2, 3, 4, 5]):
    """
    Generate a list of discrete actions for a 3D bin packing environment.
    Each action consists of a position (x, y) and a rotation type (0-5).

    Args:
        bin_width (int): Total width of the bin (X-axis).
        bin_depth (int): Total depth of the bin (Y-axis).
        rotations (list): Available rotation indices (default 0-5).

    Returns:
        list: A list of tuples in the format (x, y, rotation).
    """
    actions = []
    
    # Iterate through every possible grid cell and rotation orientation
    for x in range(bin_width):
        for y in range(bin_depth):
            for rot in rotations:
                actions.append((x, y, rot))
                
    return actions
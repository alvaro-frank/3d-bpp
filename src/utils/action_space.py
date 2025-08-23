"""
Generate a list of discrete actions for a 3D bin packing environment.
Each action consists of a position (x, y) and a rotation type (0-5).
This allows the agent to place boxes in various orientations within the bin.
"""
def generate_discrete_actions(bin_width, bin_depth, rotations=[0, 1, 2, 3, 4, 5]):
    actions = []
    for x in range(bin_width):
        for y in range(bin_depth):
            for rot in rotations:
                actions.append((x, y, rot))
    return actions
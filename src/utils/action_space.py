def generate_discrete_actions(bin_width, bin_depth, rotations=[0, 1, 2, 3]):
    actions = []
    for x in range(bin_width):
        for y in range(bin_depth):
            for rot in rotations:
                actions.append((x, y, rot))
    return actions
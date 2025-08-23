import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def plot_bin(boxes, bin_size, save_path=None, title=""):
    """
    Visualize the bin and its placed boxes in 3D.

    Each box is drawn as a colored cuboid inside the bin.
    Useful for debugging, monitoring training progress, or generating GIFs.

    Parameters:
    - boxes (list): list of Box objects (each with .position and .get_rotated_size())
    - bin_size (tuple): dimensions of the bin (width, height, depth)
    - save_path (str, optional): if provided, save the plot as an image to this path
    - title (str, optional): title to display on the figure

    Returns:
    - None (displays or saves the plot)
    """
    
    # Create a new 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    
    # Set bin boundaries (X, Y, Z axes)
    ax.set_xlim([0, bin_size[0]])
    ax.set_ylim([0, bin_size[1]])
    ax.set_zlim([0, bin_size[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Color palette for boxes (cycled if more boxes than colors)
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']

    # Draw each box in the bin
    for i, box in enumerate(boxes):
        # Extract box position and rotated dimensions
        x, y, z = box.position
        w, h, d = box.get_rotated_size()

        # Define the ranges of the cuboid (start and end along each axis)
        r = [
            [x, x + w], # X range
            [y, y + h], # Y range
            [z, z + d] # Z range
        ]
        
        # Compute the 8 corner vertices of the cuboid
        vertices = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][0]],
            [r[0][0], r[1][1], r[2][0]],
            [r[0][0], r[1][0], r[2][1]],
            [r[0][1], r[1][0], r[2][1]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][0], r[1][1], r[2][1]],
        ]

        # Define the 6 faces of the cuboid (each face is 4 vertices)
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]], # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]], # top
            [vertices[0], vertices[1], vertices[5], vertices[4]], # front
            [vertices[2], vertices[3], vertices[7], vertices[6]], # back
            [vertices[1], vertices[2], vertices[6], vertices[5]], # right
            [vertices[4], vertices[7], vertices[3], vertices[0]] # left
        ]

        # Choose a color for this box
        color = colors[i % len(colors)]
        
        # Create a 3D polygon collection for the box
        box3d = Poly3DCollection(faces, linewidths=1, edgecolors='black', alpha=0.6)
        box3d.set_facecolor(color)
        
        # Add the box to the plot
        ax.add_collection3d(box3d)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

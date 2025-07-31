import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def plot_bin(boxes, bin_size, save_path=None, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlim([0, bin_size[0]])
    ax.set_ylim([0, bin_size[1]])
    ax.set_zlim([0, bin_size[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']

    for i, box in enumerate(boxes):
        x, y, z = box.position
        w, h, d = box.get_rotated_size()

        # Create vertices for a rectangular cuboid
        r = [
            [x, x + w],
            [y, y + h],
            [z, z + d]
        ]
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

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[4], vertices[7], vertices[3], vertices[0]]
        ]

        color = colors[i % len(colors)]
        box3d = Poly3DCollection(faces, linewidths=1, edgecolors='black', alpha=0.6)
        box3d.set_facecolor(color)
        ax.add_collection3d(box3d)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

class Box:
    def __init__(self, width, height, depth, id=None):
        self.width = width
        self.height = height
        self.depth = depth
        self.position = None  # posição no bin
        self.rotation_type = 0
        self.id = id

    def get_volume(self):
        return self.width * self.height * self.depth

    def place_at(self, x, y, z):
        self.position = (x, y, z)

    def rotate(self, rotation_type):
        # 6 tipos de rotação
        w, h, d = self.width, self.height, self.depth
        if rotation_type == 0:
            return (w, h, d)
        elif rotation_type == 1:
            return (w, d, h)
        elif rotation_type == 2:
            return (h, w, d)
        elif rotation_type == 3:
            return (h, d, w)
        elif rotation_type == 4:
            return (d, w, h)
        elif rotation_type == 5:
            return (d, h, w)

    def get_rotated_size(self):
        return self.rotate(self.rotation_type)

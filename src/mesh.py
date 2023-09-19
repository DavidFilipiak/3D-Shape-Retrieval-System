
class Mesh():
    num_vertices = 0
    num_faces = 0
    class_name = ""

    def __init__(self, id):
        self.id = id

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

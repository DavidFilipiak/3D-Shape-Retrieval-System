meshes = {}
feature_list = [
    "name",
    "class_name",
    "num_vertices",
    "num_faces",
    "num_triangles",
    "num_quads",
    "bb_dim_x",
    "bb_dim_y",
    "bb_dim_z",
]

class Mesh():
    num_vertices = 0
    num_faces = 0
    num_triangles = 0
    num_quads = 0
    class_name = ""
    name = ""
    bb_dim_x = 0
    bb_dim_y = 0
    bb_dim_z = 0

    def __init__(self, id):
        self.pymeshlab_id = id

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        string = ""
        for feature in feature_list:
            string += f"{feature}: {getattr(self, feature)}\n"
        return string

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
    "volume",
    "surface_area",
    "average_edge_length",
    "total_edge_length",
    "center_of_mass",
    "connected_components_number",
    "convex_hull"

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
    volume = 0
    surface_area = 0
    average_edge_length = 0
    total_edge_length = 0

    def __init__(self, id):
        self.pymeshlab_id = id

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_features_dict(self):
        return {feature: getattr(self, feature) for feature in feature_list}

    def __str__(self):
        string = ""
        for feature in feature_list:
            string += f"{feature}: {getattr(self, feature)}\n"
        return string

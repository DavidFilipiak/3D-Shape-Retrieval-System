meshes = {}

class Mesh():
    num_vertices = 0
    num_faces = 0
    num_triangles = 0
    num_quads = 0
    class_name = ""
    name = ""

    def __init__(self, id):
        self.pymeshlab_id = id

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return \
            f"Mesh id: {self.pymeshlab_id}\n" \
            f"num_vertices: {self.num_vertices}\n" \
            f"num_faces: {self.num_faces}\n" \
            f"num_triangles: {self.num_triangles}\n" \
            f"num_quads: {self.num_quads}\n" \
            f"class_name: {self.class_name}\n" \
            f"name: {self.name}\n"

from feature import *

meshes = {}

# have separate variables for:
# 1. features - needed for querying
# 2. characteristics - needed for proofs

class Mesh():
    name = Name("")
    class_name = Class_name("")

    num_vertices = Num_vertices(0)
    num_faces = Num_faces(0)
    num_triangles = Num_triangles(0)
    num_quads = Num_quads(0)
    bb_dim_x = Bb_dim_x(0)
    bb_dim_y = Bb_dim_y(0)
    bb_dim_z = Bb_dim_z(0)
    bb_diagonal = Bb_diagonal(0)
    barycenter = Barycenter(np.zeros(3))
    major_eigenvector = MajorEigenvector(np.zeros(3))
    median_eigenvector = MedianEigenvector(np.zeros(3))
    minor_eigenvector = MinorEigenvector(np.zeros(3))
    mass_directions = MassDirections(np.zeros(3))

    volume = Volume(0)
    surface_area = Surface_area(0)
    compactness = Compactness(0)
    convexivity = Convexivity(0)
    eccentricity = Eccentricity(0)
    rectangularity = Rectangularity(0)
    diameter = Diameter(0)
    aabb_volume = AABB_volume(0)
    ch_volume = CH_volume(0)
    ch_surface_area = CH_surface_area(0)
    ch_compactness = CH_compactness(0)
    ch_eccentricity = CH_eccentricity(0)
    ch_rectangularity = CH_rectangularity(0)
    ch_diameter = CH_diameter(0)
    ch_aabb_volume = CH_AABB_volume(0)
    a3 = A3(np.zeros((2, 1)))
    d1 = D1(np.zeros((2, 1)))
    d2 = D2(np.zeros((2, 1)))
    d3 = D3(np.zeros((2, 1)))
    d4 = D4(np.zeros((2, 1)))

    def __init__(self, id):
        self.pymeshlab_id = id

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_features_dict(self):
        return {feature: getattr(self, feature) for feature in feature_list + descriptor_list + descriptor_shape_list}

    def __str__(self):
        string = ""
        for feature in feature_list:
            string += f"{feature}: {getattr(self, feature)}\n"
        for descriptor in descriptor_list:
            string += f"{descriptor}: {getattr(self, descriptor)}\n"
        return string

    def create_features(self):
        pass



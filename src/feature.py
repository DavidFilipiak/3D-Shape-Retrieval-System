import numpy as np

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
    "bb_diagonal",
    "barycenter",
    "major_eigenvector",
]

descriptor_list = [
    "volume",
    "surface_area",
    "convex_hull",
    "eccentricity"
    "compactness",
    "rectangularity",
    "diameter",
    "aabb_volume",
    "convexivity",
]

vector_feature_list = [
    "barycenter",
    "major_eigenvector",
]


class Feature():
    def __init__(self, name, min, max, value):
        self.name = name
        self.min = min
        self.max = max
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __str__(self):
        return f"{self.name}: {self.value}"


class ScalarFeature(Feature):
    def __init__(self, name, min, max, value):
        super().__init__(name, min, max, value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.min is not None and value < self.min:
            raise ValueError(f"Value {value} is less than the minimum value {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Value {value} is greater than the maximum value {self.max}")
        self._value = value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value


class VectorFeature(Feature):
    def __init__(self, name, min, max, value):
        super().__init__(name, min, max, value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.min is not None:
            for i in range(len(value)):
                if value[i] < self.min:
                    raise ValueError(f"Value {value} is less than the minimum value {self.min}")
        if self.max is not None:
            for i in range(len(value)):
                if value[i] > self.max:
                    raise ValueError(f"Value {value} is greater than the maximum value {self.max}")
        self._value = value


###############
# FEATURES
###############

class Name(Feature):
    def __init__(self, value):
        super().__init__("name", None, None, value)


class Class_name(Feature):
    def __init__(self, value):
        super().__init__("class_name", None, None, value)


class Num_vertices(ScalarFeature):
    def __init__(self, value):
        super().__init__("num_vertices", 0, 100000, value)


class Num_faces(ScalarFeature):
    def __init__(self, value):
        super().__init__("num_faces", 0, 100000, value)


class Num_triangles(ScalarFeature):
    def __init__(self, value):
        super().__init__("num_triangles", 0, 100000, value)


class Num_quads(ScalarFeature):
    def __init__(self, value):
        super().__init__("num_quads", 0, 100000, value)


class Bb_dim_x(ScalarFeature):
    def __init__(self, value):
        super().__init__("bb_dim_x", 0, 1000, value)


class Bb_dim_y(ScalarFeature):
    def __init__(self, value):
        super().__init__("bb_dim_y", 0, 1000, value)


class Bb_dim_z(ScalarFeature):
    def __init__(self, value):
        super().__init__("bb_dim_z", 0, 1000, value)


class Bb_diagonal(ScalarFeature):
    def __init__(self, value):
        super().__init__("bb_diagonal", 0, 200, value)


class Barycenter(VectorFeature):
    def __init__(self, value):
        super().__init__("barycenter", -10000000, 10000000, value)


class MajorEigenvector(VectorFeature):
    def __init__(self, value):
        super().__init__("major_eigenvector", -1.2, 1.2, value)


###############
# DESCRIPTORS
###############

class Volume(ScalarFeature):
    def __init__(self, value):
        super().__init__("volume", 0, 100000, value)


class Convex_hull(ScalarFeature):
    def __init__(self, value):
        super().__init__("convex_hull", 0, 100000, value)


class Eccentricity(ScalarFeature):
    def __init__(self, value):
        super().__init__("eccentricity", 0, 1, value)


class Rectangularity(ScalarFeature):
    def __init__(self, value):
        super().__init__("rectangularity", 0, 1, value)


class Compactness(ScalarFeature):
    def __init__(self, value):
        super().__init__("compactness", 0, 1, value)


class Convexivity(ScalarFeature):
    def __init__(self, value):
        super().__init__("convexivity", 0, 1, value)


class Surface_area(ScalarFeature):
    def __init__(self, value):
        super().__init__("surface_area", 0, 100000, value)


class Compactness(ScalarFeature):
    def __init__(self, value):
        super().__init__("connected_components_number", 0, 100000, value)


# Axis aligned bounding box
class AABB_volume(ScalarFeature):
    def __init__(self, value):
        super().__init__("obb_volume", 0, 100000, value)


class Diameter(ScalarFeature):
    def __init__(self, value):
        super().__init__("diameter", 0, 100000, value)


# this is just a dictionary of features that are displayed in the GUI and from which we grab the min and max values
show_feature_dict = {
    "num_vertices": Num_vertices(0),
    "num_faces": Num_faces(0),
    "num_triangles": Num_triangles(0),
    "num_quads": Num_quads(0),
    "bb_dim_x": Bb_dim_x(0),
    "bb_dim_y": Bb_dim_y(0),
    "bb_dim_z": Bb_dim_z(0),
    "bb_diagonal": Bb_diagonal(0),
    "barycenter": Barycenter(np.zeros(3)),
    "major_eigenvector": MajorEigenvector(np.zeros(3)),
}

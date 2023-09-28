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
    "volume",
    "surface_area",
    "average_edge_length",
    "total_edge_length",
    "connected_components_number",
    "convex_hull",
    "eccentricity"
    "compactness",
    "rectangularity"
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
        if self.min is not None and value < self.min:
            raise ValueError(f"Value {value} is less than the minimum value {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Value {value} is greater than the maximum value {self.max}")
        self._value = value
    def __str__(self):
        return f"{self.name}: {self.value}"
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
                if value[i] < self.min[i]:
                    raise ValueError(f"Value {value} is less than the minimum value {self.min}")
        if self.max is not None:
            for i in range(len(value)):
                if value[i] > self.max[i]:
                    raise ValueError(f"Value {value} is greater than the maximum value {self.max}")
        self._value = value
    def __eq__(self, other):
        return np.array_equal(self.value, other.value)
    def __lt__(self, other):
        return np.array_less(self.value, other.value)
    def __gt__(self, other):
        return np.array_greater(self.value, other.value)
    def __le__(self, other):
        return np.array_less_equal(self.value, other.value)
    def __ge__(self, other):
        return np.array_greater_equal(self.value, other.value)

class Name(Feature):
    def __init__(self, value):
        super().__init__("name", None, None, value)

class Class_name(Feature):
    def __init__(self, value):
        super().__init__("class_name", None, None, value)

class Num_vertices(Feature):
    def __init__(self, value):
        super().__init__("num_vertices", 0, 100000, value)

class Num_faces(Feature):
    def __init__(self, value):
        super().__init__("num_faces", 0, 100000, value)

class Num_triangles(Feature):
    def __init__(self, value):
        super().__init__("num_triangles", 0, 100000, value)

class Num_quads(Feature):
    def __init__(self, value):
        super().__init__("num_quads", 0, 100000, value)

class Bb_dim_x(Feature):
    def __init__(self, value):
        super().__init__("bb_dim_x", 0, 100000, value)

class Bb_dim_y(Feature):
    def __init__(self, value):
        super().__init__("bb_dim_y", 0, 100000, value)

class Bb_dim_z(Feature):
    def __init__(self, value):
        super().__init__("bb_dim_z", 0, 100000, value)

class Bb_diagonal(Feature):
    def __init__(self, value):
        super().__init__("bb_diagonal", 0, 100000, value)

class Barycenter(VectorFeature):
    def __init__(self, value):
        super().__init__("barycenter", np.ones(3)*-10000000, np.ones(3)*10000000, value)

class Volume(Feature):
    def __init__(self, value):
        super().__init__("volume", 0, 100000, value)

class Surface_area(Feature):
    def __init__(self, value):
        super().__init__("surface_area", 0, 100000, value)

class Average_edge_length(Feature):
    def __init__(self, value):
        super().__init__("average_edge_length", 0, 100000, value)

class Total_edge_length(Feature):
    def __init__(self, value):
        super().__init__("total_edge_length", 0, 100000, value)

class Connected_components_number(Feature):
    def __init__(self, value):
        super().__init__("connected_components_number", 0, 100000, value)

class Convex_hull(Feature):
    def __init__(self, value):
        super().__init__("convex_hull", 0, 100000, value)

class Eccentricity(Feature):
    def __init__(self, value):
        super().__init__("eccentricity", 0, 1, value)

class Rectangularity(Feature):
    def __init__(self, value):
        super().__init__("rectangularity", 0, 1, value)

class Compactness(Feature):
    def __init__(self, value):
        super().__init__("compactness", 0, 1, value)

class Convexivity(Feature):
    def __init__(self, value):
        super().__init__("convexivity", 0, 1, value)
#Axis aligned bounding box
class AABB_volume(Feature):
    def __init__(self, value):
        super().__init__("obb_volume", 0, 100000, value)
class Diameter(Feature):
    def __init__(self, value):
        super().__init__("diameter", 0, 100000, value)




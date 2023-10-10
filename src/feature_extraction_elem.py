import math
import pymeshlab
from utils import *
import numpy as np
from mesh import Mesh

#feature extraction chaining
def get_elementary_features(mesh:Mesh, meshSet:pymeshlab.MeshSet) ->Mesh:
    print(meshSet.current_mesh().id())
    mesh.set_params(
        volume= get_volume(meshSet.current_mesh()),
        surface_area= get_surface_area(meshSet),
        compactness=get_compactness(meshSet.current_mesh(), meshSet),
        #convex_hull=get_convex_hull_volume(meshSet),
        eccentricity=get_eccentricity(meshSet.current_mesh()),
        rectangularity=get_rectangularity(meshSet.current_mesh()),
        diameter=get_diameter(meshSet.current_mesh()),
        aabb_volume=get_AABB_volume(meshSet.current_mesh())
    )
    #extract convex hull features
    meshSet.generate_convex_hull()
    print(f"convex hull id:{meshSet.current_mesh().id()}")
    mesh.set_params(
        ch_volume=get_volume(meshSet.current_mesh()),
        ch_surface_area=get_surface_area(meshSet),
        ch_compactness=get_compactness(meshSet.current_mesh(), meshSet),
        ch_convexivity=get_convexivity(meshSet.current_mesh()),
        ch_eccentricity=get_eccentricity(meshSet.current_mesh()),
        ch_rectangularity=get_rectangularity(meshSet.current_mesh()),
        ch_diameter=get_diameter(meshSet.current_mesh()),
        ch_aabb_volume=get_AABB_volume(meshSet.current_mesh())
    )
    meshSet.delete_current_mesh()
    return mesh

# def get_elementary_features_convex_hull(mesh:Mesh, meshSet:pymeshlab.MeshSet) ->Mesh:
#     print(meshSet.current_mesh().id())
#     meshSet.generate_convex_hull()
#     print(f"convex hull id:{meshSet.current_mesh().id()}")
#     mesh.set_params(
#         ch_volume= get_volume(meshSet.current_mesh()),
#         ch_surface_area = get_surface_area(meshSet),
#         ch_compactness = get_compactness(meshSet.current_mesh(), meshSet),
#         ch_convexivity = get_convexivity(meshSet.current_mesh()),
#         ch_eccentricity = get_eccentricity(meshSet.current_mesh()),
#         ch_rectangularity = get_rectangularity(meshSet.current_mesh()),
#         ch_diameter = get_diameter(meshSet.current_mesh()),
#         ch_aabb_volume = get_AABB_volume(meshSet.current_mesh())
#     )
#     meshSet.delete_current_mesh()
#     return mesh
def get_surface_area(meshSet:pymeshlab.MeshSet):
    return meshSet.get_geometric_measures()['surface_area']

def get_volume(mesh):
    return calculate_volume(mesh.vertex_matrix(), mesh.face_matrix())

def get_compactness(mesh, meshSet:pymeshlab.MeshSet):
    return (36 * math.pi * get_volume(mesh) ** 2) ** (1 / 3) / get_surface_area(meshSet)

def get_AABB_volume(mesh):
    return mesh.bounding_box().dim_x() * mesh.bounding_box().dim_y() * mesh.bounding_box().dim_z()

def get_convex_hull_volume(meshSet):
    return meshSet.convex_hull()
def get_eccentricity(mesh):
    scale_long = max(mesh.bounding_box().dim_x(), mesh.bounding_box().dim_y(), mesh.bounding_box().dim_z())
    scale_min = min(mesh.bounding_box().dim_x(), mesh.bounding_box().dim_y(), mesh.bounding_box().dim_z())
    return math.sqrt(1 - (scale_min / scale_long) ** 2)

def get_rectangularity(mesh):
    return get_AABB_volume(mesh) / get_volume(mesh)


def get_convexivity(mesh):
    return get_volume(mesh) / get_AABB_volume(mesh)

def get_diameter(mesh):
    distances = np.linalg.norm(mesh.vertex_matrix() - mesh.vertex_matrix().mean(axis=0), axis=1)
    return distances.max()
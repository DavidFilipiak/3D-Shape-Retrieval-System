import math
import pymeshlab
from utils import *
import numpy as np
from mesh import Mesh
from scipy.spatial import distance

#feature extraction chaining
def get_elementary_features(mesh:Mesh, meshSet:pymeshlab.MeshSet) ->Mesh:
    print(meshSet.current_mesh().id())
    mesh.set_params(
        volume= get_volume(meshSet.current_mesh()),
        surface_area= get_surface_area(meshSet.current_mesh()),
        compactness=get_compactness(meshSet.current_mesh(), meshSet),
        eccentricity=get_eccentricity(meshSet.current_mesh()),
        rectangularity=get_rectangularity(meshSet.current_mesh()),
        diameter=get_diameter(meshSet.current_mesh()),
        aabb_volume=get_AABB_volume(meshSet.current_mesh())
    )
    #extract convex hull features
    meshSet.generate_convex_hull()
    #meshSet.save_current_mesh("/Users/georgioschristopoulos/PycharmProjects/3D-Shape-Retrieval-System/convex_hull.obj")
    print(f"convex hull id:{meshSet.current_mesh().id()}")
    mesh.set_params(
        ch_volume=get_volume(meshSet.current_mesh()),
        ch_surface_area=get_surface_area(meshSet.current_mesh()),
        ch_compactness=get_compactness(meshSet.current_mesh(), meshSet),
        ch_eccentricity=get_eccentricity(meshSet.current_mesh()),
        ch_rectangularity=get_rectangularity(meshSet.current_mesh()),
        ch_diameter=get_diameter(meshSet.current_mesh()),
        ch_aabb_volume=get_AABB_volume(meshSet.current_mesh())

    )
    convexivity = mesh.volume/ mesh.ch_volume
    mesh.set_params(
        convexivity=convexivity
    )
    meshSet.delete_current_mesh()
    return mesh

def get_surface_area(mesh):
    total_face_area = calculate_face_area(mesh.face_matrix(), mesh.vertex_matrix())
    return sum(total_face_area)

def get_volume(mesh):
    return calculate_volume(mesh.vertex_matrix(), mesh.face_matrix())

def get_compactness(mesh, meshSet:pymeshlab.MeshSet):
    S = get_surface_area(meshSet.current_mesh())
    return S ** 3 / (36 * math.pi * get_volume(mesh) ** 2)

def get_AABB_volume(mesh):
    return mesh.bounding_box().dim_x() * mesh.bounding_box().dim_y() * mesh.bounding_box().dim_z()

def get_eccentricity(mesh):
    principal_components = get_principal_components(mesh.vertex_matrix())
    major_eigenvalue = principal_components[0][0]
    minor_eigenvalue = principal_components[2][0]
    eccentricity = abs(major_eigenvalue) / abs(minor_eigenvalue)
    return eccentricity

def get_rectangularity(mesh):
    return get_volume(mesh)/get_AABB_volume(mesh)


def get_convexivity(mesh):
    return mesh.volume / mesh.ch_volume

def get_diameter(mesh):
    vertices = mesh.vertex_matrix()
    distances = distance.cdist(vertices, vertices, 'euclidean')
    return distances.max()

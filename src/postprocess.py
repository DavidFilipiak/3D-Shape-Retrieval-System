import math
import pymeshlab
from utils import *
import numpy as np
from mesh import Mesh


def stitch_holes(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    meshSet.meshing_close_holes(maxholesize=2000, selfintersection=False)

    current_mesh = meshSet.current_mesh()
    num_triangles, num_quads = count_triangles_and_quads(current_mesh.polygonal_face_list())

    mesh.set_params(
        num_vertices = current_mesh.vertex_number(),
        num_faces = current_mesh.face_number(),
        num_triangles = num_triangles,
        num_quads = num_quads
    )
    return mesh


def fix_face_normals(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    face_normal_matrix = meshSet.current_mesh().face_normal_matrix()



    return mesh
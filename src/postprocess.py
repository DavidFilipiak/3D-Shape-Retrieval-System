import math
import pymeshlab
from pymeshlab import Percentage

from utils import *
import numpy as np
from mesh import Mesh


def stitch_holes(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    if (meshSet.get_topological_measures()["number_holes"] !=0):
        if (meshSet.get_topological_measures()["non_two_manifold_edges"] > 0):
            meshSet.meshing_repair_non_manifold_edges(method = 1)
        meshSet.meshing_close_holes(maxholesize=2000, selfintersection=True)
        #meshSet.meshing_merge_close_vertices(threshold= Percentage(1))
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
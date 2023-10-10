import math
import pymeshlab
from utils import *
import numpy as np
from mesh import Mesh
import random

# angle between three random points
def a3(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    vertex_list = meshSet.current_mesh().vertex_matrix()
    angles = []
    N = len(vertex_list)
    n = 100000
    print(N, n)
    k = int(n ** (1/3))
    for i in range(0, k):
        v1 = random.randint(0, N - 1)
        for j in range(0, k):
            v2 = random.randint(0, N - 1)
            if v1 == v2:
                continue
            for l in range(0, k):
                v3 = random.randint(0, N - 1)
                if v1 == v3 or v2 == v3:
                    continue
                _v1, _v2, _v3 = vertex_list[v1], vertex_list[v2], vertex_list[v3]
                vertex1 = _v2 - _v1
                vertex2 = _v3 - _v1
                angles.append(np.arccos(dot(vertex1, vertex2) / (np.linalg.norm(vertex1) * np.linalg.norm(vertex2))))

    hist_y, hist_x = np.histogram(angles, math.ceil(math.sqrt(len(angles))))
    hist_x = hist_x[:-1]
    mesh.set_params(
        a3 = np.array([hist_x, hist_y])
    )
    return mesh


# distance between a random point and a barycenter
def d1(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    vertex_list = meshSet.current_mesh().vertex_matrix()
    dists = np.linalg.norm(vertex_list, axis=1)
    hist_y, hist_x = np.histogram(dists, math.ceil(math.sqrt(len(dists))))
    hist_x = hist_x[:-1]
    mesh.set_params(
        d1 = np.array([hist_x, hist_y])
    )
    return mesh


# distance between two random points
def d2(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    vertex_list = meshSet.current_mesh().vertex_matrix()
    dists = []
    N = len(vertex_list)
    n = 100000
    k = int(n ** (1/2))
    for i in range(0, k):
        v1 = random.randint(0, N - 1)
        for j in range(0, k):
            v2 = random.randint(0, N - 1)
            if v1 == v2:
                continue
            dists.append(np.linalg.norm(vertex_list[v1] - vertex_list[v2]))

    hist_y, hist_x = np.histogram(dists, math.ceil(math.sqrt(len(dists))))
    hist_x = hist_x[:-1]
    mesh.set_params(
        d2 = np.array([hist_x, hist_y])
    )
    return mesh


# area of a triangle made by three random points
def d3(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    vertex_list = meshSet.current_mesh().vertex_matrix()
    areas = []
    N = len(vertex_list)
    n = 100000
    print(N, n)
    k = int(n ** (1/3))
    for i in range(0, k):
        v1 = random.randint(0, N - 1)
        for j in range(0, k):
            v2 = random.randint(0, N - 1)
            if v1 == v2:
                continue
            for l in range(0, k):
                v3 = random.randint(0, N - 1)
                if v1 == v3 or v2 == v3:
                    continue
                _v1, _v2, _v3 = vertex_list[v1], vertex_list[v2], vertex_list[v3]
                areas.append(np.linalg.norm(np.cross(_v2 - _v1, _v3 - _v1)) / 2)

    hist_y, hist_x = np.histogram(areas, math.ceil(math.sqrt(len(areas))))
    hist_x = hist_x[:-1]
    mesh.set_params(
        d3 = np.array([hist_x, hist_y])
    )
    return mesh


# volume of a tetrahedron made by four random points
def d4(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    vertex_list = meshSet.current_mesh().vertex_matrix()
    volumes = []
    N = len(vertex_list)
    n = 100000
    print(N, n)
    k = int(n ** (1/4))
    for i in range(0, k):
        v1 = random.randint(0, N - 1)
        for j in range(0, k):
            v2 = random.randint(0, N - 1)
            if v1 == v2:
                continue
            for l in range(0, k):
                v3 = random.randint(0, N - 1)
                if v1 == v3 or v2 == v3:
                    continue
                for m in range(0, k):
                    v4 = random.randint(0, N - 1)
                    if v1 == v4 or v2 == v4 or v3 == v4:
                        continue
                    _v1, _v2, _v3, _v4 = vertex_list[v1], vertex_list[v2], vertex_list[v3], vertex_list[v4]
                    volumes.append(abs(dot(_v4 - _v1, np.cross(_v2 - _v1, _v3 - _v1))) / 6)

    hist_y, hist_x = np.histogram(volumes, math.ceil(math.sqrt(len(volumes))))
    hist_x = hist_x[:-1]
    mesh.set_params(
        d4 = np.array([hist_x, hist_y])
    )
    return mesh
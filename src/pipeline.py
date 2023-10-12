from __future__ import annotations
import pymeshlab
from mesh import Mesh


class Pipeline:
    def __init__(self, meshSet: pymeshlab.MeshSet):
        self.ms = meshSet
        self.pipeline = []
        self.func_args = {}

    def add(self, modifier, **kwargs) -> None:
        self.pipeline.append(modifier)
        self.func_args[modifier] = kwargs

    def run(self, mesh: Mesh | list[Mesh], verbose=True) -> Mesh | list[Mesh]:
        if isinstance(mesh, Mesh):
            self.ms.set_current_mesh(mesh.pymeshlab_id)
            for modifier in self.pipeline:
                if verbose:
                    print("Performing " + modifier.__name__ + " on mesh " + str(mesh.pymeshlab_id) + " " + mesh.name)
                func_args = self.func_args[modifier]
                if not func_args:
                    mesh = modifier(mesh, self.ms)
                else:
                    mesh = modifier(mesh, self.ms, **func_args)
            return mesh
        elif isinstance(mesh, list):
            for modifier in self.pipeline:
                for i, m in enumerate(mesh):
                    if verbose:
                        print("Performing " + modifier.__name__ + " on mesh " + str(m.pymeshlab_id) + " " + m.name)
                    self.ms.set_current_mesh(m.pymeshlab_id)
                    func_args = self.func_args[modifier]
                    if not func_args:
                        mesh[i] = modifier(m, self.ms)
                    else:
                        mesh[i] = modifier(m, self.ms, **func_args)
            return mesh
        else:
            raise Exception("mesh must be Mesh or list[Mesh] and is " + str(type(mesh)))


import pymeshlab
from mesh import Mesh, meshes


def normalize(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:

    # apply filters
    meshSet.compute_matrix_from_translation(traslmethod='Center on Layer BBox', alllayers=True)
    meshSet.compute_matrix_from_scaling_or_normalization(unitflag=True, alllayers=True, scalecenter='barycenter')

    # change parameters of current mesh
    current_mesh = meshSet.current_mesh()
    mesh.set_params(
        bb_dim_x=current_mesh.bounding_box().dim_x(),
        bb_dim_y=current_mesh.bounding_box().dim_y(),
        bb_dim_z=current_mesh.bounding_box().dim_z(),
    )
    return mesh

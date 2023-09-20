import pymeshlab
from mesh import Mesh, meshes

def normalize(meshSet: pymeshlab.MeshSet) -> None:
    '''
    meshSet.compute_matrix_from_translation(traslmethod='Center on Layer BBox', alllayers=True)
    meshSet.compute_matrix_from_scaling_or_normalization(unitflag=True, alllayers=True, scalecenter='barycenter')
    meshSet.save_filter_script('filters/normalize_mesh.mlx')
    meshSet.load_filter_script('filters/normalize_mesh.mlx')
    meshSet.apply_filter_script()
    meshSet.save_current_mesh('normalized_mesh.obj')
    '''


    pass
from tkinter import *
from tkinter import filedialog
import os
import pymeshlab
import pandas as pd
import polyscope as ps
from mesh import Mesh, meshes
from database import Database
from matplotlib import pyplot as plt

# GLOBAL VARIABLES
ms = None
listbox_loaded_meshes = None
current_dir = os.getcwd()
selected_x = [0, 1000]  # example data for now, to store the list of values for x-axis
selected_y = [0, 2000]  # to store the list of values for y-axis

def count_triangles_and_quads(polygonal_face_list):
    num_triangles = 0
    num_quads = 0

    for face in polygonal_face_list:
        num_vertices = len(face)
        if num_vertices == 3:
            num_triangles += 1
        elif num_vertices == 4:
            num_quads += 1
    return num_triangles, num_quads


def browse_button() -> None:
    db_dir = os.path.abspath(os.path.join(current_dir, "..", "db"))
    filename = filedialog.askopenfilename(title="Mesh select", initialdir=db_dir, filetypes=[('Mesh files', '*.obj')])
    ms.load_new_mesh(filename)

    current_mesh = ms.current_mesh()
    mesh_name = "/".join(filename.split("/")[-2:])
    listbox_loaded_meshes.insert(END, mesh_name)
    num_triangles, num_quads = count_triangles_and_quads(current_mesh.polygonal_face_list())
    mesh = Mesh(current_mesh.id())
    mesh.set_params(
      num_vertices=current_mesh.vertex_number(),
      num_faces=current_mesh.face_number(),
      num_triangles=num_triangles,
      num_quads=num_quads,
      class_name=os.path.dirname(filename).split('/')[-1],
      name=mesh_name
    )
    meshes[mesh.name] = mesh
    print(mesh)

    '''
    classType = os.path.dirname(filename).split('/')[-1]
    print(f"class type: {classType}")
    print(f"face number: {ms.current_mesh().face_number()}")
    print(f"The mesh is alligned on the x-axis: {ms.current_mesh().bounding_box().dim_x()}, y-axis: {ms.current_mesh().bounding_box().dim_y()}"
          f" and z-axis: {ms.current_mesh().bounding_box().dim_z()}")
    print(f" vertex number{ms.current_mesh().vertex_number()}")
    listbox_loaded_meshes.insert(END, "/".join(filename.split("/")[-2:]))
    num_triangles, num_quads = count_triangles_and_quads(ms.current_mesh().polygonal_face_list())
    print(f"The number of triangles is {num_triangles}, and the number of quads is: {num_quads}")
  '''


# right now this function only loads custom features from the csv_files file until real ones will go there
def analyze_meshes() -> None:
    database = Database()
    database.load_tables("csv_files")
    features_table = database.get_table("features")
    print(features_table)
    # TODO: Jesse --> analyze and print requested features, use function from pandas when possible
    analyze_df(features_table)


def analyze_df(df) -> None:
    for f in df:
        feature = pd.Series([f]).describe()  # More descriptive statistics should go in here
        print(feature)


def draw_histogram(arr_x, arr_y):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.bar(arr_x, arr_y)
    plt.plot(arr_x, arr_y)
    return fig


def main() -> None:
    global ms, listbox_loaded_meshes
    ms = pymeshlab.MeshSet()

    root = Tk()
    root.title("3D Shape Retrieval")
    root.geometry("500x500")

    button_browse = Button(text="Load Mesh", command=browse_button)
    button_browse.grid(row=0, column=1)
    button_show = Button(text="Show Loaded Meshes", command=ms.show_polyscope)
    button_show.grid(row=0, column=2)
    button_analyze = Button(text="Analyze", command=analyze_meshes)
    button_analyze.grid(row=0, column=3)

    label_loaded_meshes = Label(root, text="Loaded Meshes")
    label_loaded_meshes.grid(row=1, column=1)
    listbox_loaded_meshes = Listbox(root, width=50)
    listbox_loaded_meshes.grid(row=2, column=1, columnspan=3)

    button_graph = Button(root, text="Show histogram", command=draw_histogram(selected_x, selected_y))
    button_graph.grid(row=3, column=1)

    # class_type_label = Label(root, text="Class type: N/A")
    # class_type_label.grid(row=1, column=1)
    #
    # face_number_label = Label(root, text="Face number: N/A")
    # face_number_label.grid(row=2, column=1)
    #
    # vertex_number_label = Label(root, text="Vertex number: N/A")
    # vertex_number_label.grid(row=3, column=1)

    root.mainloop()


if __name__ == "__main__":
    main()
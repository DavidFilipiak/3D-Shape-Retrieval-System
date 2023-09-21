from tkinter import *
from tkinter import filedialog
import os
import pymeshlab
import pandas as pd
import polyscope as ps
from mesh import Mesh, meshes
from database import Database
from matplotlib import pyplot as plt
from preprocess import normalize
from utils import count_triangles_and_quads
from pipeline import Pipeline

# GLOBAL VARIABLES
ms = None
listbox_loaded_meshes = None
current_dir = os.getcwd()
selected_x = [0, 1000]  # example data for now, to store the list of values for x-axis
selected_y = [0, 2000]  # to store the list of values for y-axis
curr_mesh = None


def resample_mesh(mesh, vertex_num, face_num,filename) -> None:
  if (vertex_num < 100 or face_num < 100):
    print("The mesh is poorly-sampled.")
    mesh.subdivision_surfaces_midpoint()
    filename_save = filename.split("/")[-1].split('.')[0]
    last_slash_index = filename.rfind('/')
    result_path = filename[:last_slash_index]
    ms.save_current_mesh(os.path.join(result_path,filename_save+"_resampled.obj"))

def browse_button() -> None:
    global curr_mesh

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
        name=mesh_name,
        bb_dim_x=current_mesh.bounding_box().dim_x(),
        bb_dim_y=current_mesh.bounding_box().dim_y(),
        bb_dim_z=current_mesh.bounding_box().dim_z(),
    )
    meshes[mesh.name] = mesh
    curr_mesh = mesh
    print(mesh)

    #resample_mesh(ms,mesh.num_vertices, mesh.num_faces, filename)
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

def load_all_meshes() -> None:
    folder_name = filedialog.askdirectory(title="Mesh select", initialdir=os.path.abspath(os.path.join(current_dir, "..", "db")))
    for class_name in os.listdir(folder_name):
        for filename in os.listdir(os.path.join(folder_name, class_name)):
            if filename.endswith(".obj"):
                ms.load_new_mesh(os.path.join(folder_name, class_name, filename))
                listbox_loaded_meshes.insert(END, f"{class_name}/{filename}")

def show():
    ms.show_polyscope()


def normalize_btn():
    global ms, curr_mesh
    p = Pipeline(ms)
    p.add(normalize)
    curr_mesh = p.run(curr_mesh)


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


def do_nothing():
    pass

def main() -> None:
    global ms, listbox_loaded_meshes
    ms = pymeshlab.MeshSet()

    root = Tk()
    root.title("3D Shape Retrieval")
    root.geometry("500x500")

    menubar = Menu(root)
    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label="Load Mesh", command=browse_button)
    filemenu.add_command(label="Load All (.obj)", command=browse_button)
    filemenu.add_command(label="Load All (.csv)", command=do_nothing)

    button_browse = Button(text="Load Mesh", command=browse_button)
    button_browse.grid(row=0, column=1)
    button_show = Button(text="Show Loaded Meshes", command=show)
    button_show.grid(row=0, column=2)
    button_analyze = Button(text="Analyze", command=analyze_meshes)
    button_analyze.grid(row=0, column=3)
    button_normalize = Button(text="Normalize", command=normalize_btn)
    button_normalize.grid(row=0, column=4)

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
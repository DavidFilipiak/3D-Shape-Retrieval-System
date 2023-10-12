import pandas as pd
from database import Database

database = None
def main()->None:
    counter_surface_area = 0
    counter_volume = 0
    database = Database()
    filename = "/Users/georgioschristopoulos/PycharmProjects/3D-Shape-Retrieval-System/all_desc.csv"
    database.load_table(filename)
    df = database.get_table()
    # Iterate over the rows of the dataframe and call the assert function for each row.
    for row in df.iterrows():
      mesh_info = row[1]
      # assert mesh_info["volume"] <= mesh_info["ch_volume"], (
      #     "The volume of the mesh ({}) is not smaller than the volume of its convex hull".format(
      #         mesh_info["name"]))
      # # assert mesh_info["diameter"] <= mesh_info["ch_diameter"], (
      # #     "The diameter of the mesh ({}) is not smaller than the volume of its convex hull".format(
      # #         mesh_info["name"]))
      # assert mesh_info["surface_area"] <= mesh_info["ch_surface_area"], (
      #     "The surface area of the mesh ({}) is not smaller than the surface area of its convex hull".format(
      #         mesh_info["name"]))
      if(mesh_info["volume"] > mesh_info["ch_volume"]):
          counter_volume += 1
          print("The volume of the mesh ({}) is not smaller than the volume of its convex hull".format(
                   mesh_info["name"]))
      if(mesh_info["surface_area"] > mesh_info["ch_surface_area"]):
          counter_surface_area += 1
          print("The surface area of the mesh ({}) is not smaller than the surface area of its convex hull".format(
                mesh_info["name"]))
    print("volume counter: {}".format(counter_volume))
    print("surface area counter: {}".format(counter_surface_area))
if __name__ == "__main__":
    main()
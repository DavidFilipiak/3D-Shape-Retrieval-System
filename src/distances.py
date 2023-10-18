import os

import numpy as np
from src.main import current_dir
from scipy.stats import wasserstein_distance
import pandas as pd
from database import Database
from sklearn.preprocessing import StandardScaler




def calculate_euclidean_distances(df,object_id):
    # Select the features of the object with the given ID


    # # Create a StandardScaler for standardization
    # scaler = StandardScaler()
    #
    # # Normalize the single-value features (assuming columns a1 to a5 are single-value features)
    # df[['volume', 'surface_area', 'eccentricity', 'rectangularity', 'diameter', 'aabb_volume', 'convexivity']] = scaler.fit_transform(df[['volume', 'surface_area', 'eccentricity', 'rectangularity', 'diameter', 'aabb_volume', 'convexivity']])
    #
    #
    # # Normalize the histogram features (assuming columns a6 to a45 are histogram features)
    # df[['a3','d1','d2','d3'	,'d4']] = df[['a3','d1','d2','d3','d4']] / df[['a6', 'a7', ..., 'a15']].sum(axis=1)

    distances_df = pd.DataFrame(columns=['Name', 'Eucl_Distance'])
    object_features = df.loc[df['name'] == object_id].iloc[:, 2:].values
    for index, row in df.iterrows():
        distance = get_euclidean_distance(row.values[2:], object_features[0], 0, 1)
        distances_df = distances_df.append({'Name': row.values[0], 'Eucl_Distance': distance},ignore_index=True)
    # Calculate the Euclidean distances for all objects

    # Sort the DataFrame by distance in ascending order
    sorted_distances = distances_df.sort_values(by='Eucl_Distance')
    # Get the top 10 most similar objects
    top_10_similar = sorted_distances.head(10)
    return sorted_distances

def get_euclidean_distance(vec_a, vec_b, range_min, range_max, normalize=True):
    dist = np.linalg.norm(vec_a - vec_b)
    if normalize:
        max_dist = np.sqrt(len(vec_a) * ((range_max - range_min) ** 2))
        dist /= max_dist

    return dist




def get_emd(df, query_object):
    distances_df = pd.DataFrame(columns=['Name', 'Eucl_Distance'])
    object_features = df.loc[df['name'] == query_object].iloc[:, 2:].values[0]
    query_data = []
    for i in object_features:
        query_data.append(i[1])
    #pad = np.zeros(50)
   # query_data[1] = np.concatenate((query_data[1], pad), axis=None)
    for index, row in df.iterrows():
        data = []
        for i in row.values[2:]:
            data.append(i[1])
        #data[1] = np.concatenate((data[1], pad), axis=None)
        overall_distance = 0
        for i in range(0, len(data)):
            distance = get_emd_distance(data[i],query_data[i])
            overall_distance += distance
        distances_df = distances_df.append({'Name': row.values[0], 'EMD_Distance': overall_distance},ignore_index=True)

    return distances_df.sort_values(by='EMD_Distance')

def get_emd_distance(vec_a, vec_b):
    return wasserstein_distance(vec_a, vec_b)

def main()->None:
    database = Database()
    filename = os.path.abspath(os.path.join(current_dir, "csv_files", "aaaaaaaaaaall_desc.csv"))
    database.load_table(filename)
    df = database.get_table()
    filename = os.path.abspath(os.path.join(current_dir, "csv_files", "shape_descriptors_final_standardized.csv"))
    database.clear_table()
    database.load_table(filename)
    df2 = database.get_table()
    result = pd.merge(df, df2, on='name', how='outer')
    #similar_objects_eucl_df = calculate_euclidean_distances(df, "AircraftBuoyant/m1337.obj")

    similar_objects_emd_df = get_emd(df2, "AircraftBuoyant/m1337.obj")
    print("FINISHED")
    # Display the result
if __name__ == "__main__":
    main()

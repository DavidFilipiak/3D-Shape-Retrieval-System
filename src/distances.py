import os

import numpy as np
from src.main import current_dir
from scipy.stats import wasserstein_distance
import pandas as pd
from database import Database
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def calculate_euclidean_distances(df,object_id):
    #assign weights to the dataframe
    weights = [0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15]
    df.iloc[:, 2:] = df.iloc[:, 2:].mul(weights)
    # Create a new DataFrame to store the distances
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
    #assign weights to the dataframe
    weights = [0.3, 0.1, 0.2, 0.3, 0.1]
    df.iloc[:, 2:] = df.iloc[:, 2:].mul(weights)
    distances_df = pd.DataFrame(columns=['Name', 'a3_distance', 'd1', 'd2', 'd3', 'd4'])
    object_features = df.loc[df['name'] == query_object].iloc[:, 2:].values[0]
    query_data = []
    for i in object_features:
        query_data.append(i[1])
    for index, row in df.iterrows():
        data = []
        for i in row.values[2:]:
            data.append(i[1])
        for i in range(0, len(data)):
            distance = get_emd_distance(data[i],query_data[i])
            #overall_distance += distance
            if i == 0:
                distances_df = distances_df.append({'Name': row.values[0], 'a3_distance': distance},ignore_index=True)
            else:
                distances_df['d'+str(i)]= distance
    return distances_df

def get_emd_distance(vec_a, vec_b):
    return wasserstein_distance(vec_a, vec_b)

def main()->None:
    database = Database()
    filename = os.path.abspath(os.path.join(current_dir, "csv_files", "aaaaaaaaaaall_desc_standardized.csv"))
    database.load_table(filename)
    df = database.get_table()
    filename = os.path.abspath(os.path.join(current_dir, "csv_files", "shape_descriptors_final_standardized.csv"))
    database.clear_table()
    database.load_table(filename)
    df2 = database.get_table()
    result = pd.merge(df, df2, on='name', how='inner')
    similar_objects_eucl_df = calculate_euclidean_distances(df, "AircraftBuoyant/m1337.obj")

    similar_objects_emd_df = get_emd(df2, "AircraftBuoyant/m1337.obj")
    print("FINISHED")
    # Display the result
if __name__ == "__main__":
    main()

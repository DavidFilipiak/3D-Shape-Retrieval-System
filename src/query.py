import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from feature import descriptor_list, descriptor_shape_list
from utils import flatten_nested_array


elem_weights = [0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15]
hist_weights = [0.3, 0.1, 0.2, 0.3, 0.1]
common_weights = [0.4, 0.6]   # elem, hist with respect to each other

def calculate_euclidean_distances(df, object_id):
    #assign weights to the dataframe
    #df.iloc[:, 2:] = df.iloc[:, 2:].mul(elem_weights)

    # Create a new DataFrame to store the distances
    distances_df = pd.DataFrame(columns=['Name', 'Eucl_Distance'])
    object_features = df.loc[df['name'] == object_id].iloc[:, 2:].values[0]
    object_features = flatten_nested_array(object_features)
    #distances_df.loc[:["Eucl_Distance"]] = df.iloc[:, 2:].apply(lambda x: get_euclidean_distance(x.values, object_features[0], 0, 1, False), axis=1)
    for index, row in df.iterrows():
        new_features = flatten_nested_array(row.values[2:])
        if len(new_features) > 400:
            continue
        distance = lp_norm(new_features, object_features)
        #distance = get_cosine_sim(new_features, object_features)
        distances_df = pd.concat([distances_df, pd.DataFrame({'Name': [row.values[0]], 'Eucl_Distance': [distance]})], axis=0, ignore_index=True)
    # Calculate the Euclidean distances for all objects

    # Sort the DataFrame by distance in ascending order
    sorted_distances = distances_df.sort_values(by='Eucl_Distance')
    # Get the top 10 most similar objects
    top_10_similar = sorted_distances.head(10)
    return sorted_distances

def get_cosine_sim(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def lp_norm(vec_a, vec_b, p=1):
    dists = np.abs(vec_a - vec_b)
    sum = np.sum(dists ** p)
    return sum ** (1 / p)

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
                distances_df = pd.concat([distances_df, pd.DataFrame({'Name': [row.values[0]], 'a3_distance': [distance]})], axis=0, ignore_index=True)
            else:
                distances_df['d'+str(i)]= distance
    return distances_df

def get_emd_distance(vec_a, vec_b):
    return wasserstein_distance(vec_a, vec_b)


def naive_weighted_features(mesh_name, df_elem, df_hist, n=5):
    # weight elementary features
    df_elem.iloc[:, 2:] = df_elem.iloc[:, 2:].mul(elem_weights).mul(common_weights[0])

    # weight histogram features
    for i, col in enumerate(descriptor_shape_list):
        df_hist.loc[:, col] = df_hist.loc[:, col].apply(lambda x: x[1] * hist_weights[i] * common_weights[1])

    df_together = pd.merge(df_elem, df_hist, on=['name', 'class_name'], how='inner')
    eucl_dist_df = calculate_euclidean_distances(df_together, mesh_name)
    return eucl_dist_df.head(n + 1).values



def naive_weighted_distances(mesh_name, df, n=5):
    df = df.copy()
    print(df.head())
    df_filter = df[df['name1'] == mesh_name]
    df_filter = df_filter.sort_values(by='dist')
    df_filter = df_filter.head(n + 1)
    return df_filter[['name2', 'dist']].values
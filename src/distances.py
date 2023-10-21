import os

import numpy as np
from src.main import current_dir
from scipy.stats import wasserstein_distance
import pandas as pd
from database import Database
from feature import descriptor_list, descriptor_shape_list


elem_weights = [0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15]
hist_weights = [0.3, 0.1, 0.2, 0.3, 0.1]
common_weights = [0.6, 0.4]   # elem, hist with respect to each other


def calculate_euclidean_distances(df,object_id):
    #assign weights to the dataframe
    weights = [0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15,1.5714285714/15]
    df.iloc[:, 2:] = df.iloc[:, 2:].mul(weights)
    # Create a new DataFrame to store the distances
    distances_df = pd.DataFrame(columns=['Name', 'Eucl_Distance'])
    object_features = df.loc[df['name'] == object_id].iloc[:, 2:].values
    for index, row in df.iterrows():
        distance = get_euclidean_distance(row.values[2:], object_features[0], 0, 1)
        distances_df = pd.concat([distances_df, pd.DataFrame({'Name': [row.values[0]], 'Eucl_Distance': [distance]})], axis=0, ignore_index=True)
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
                distances_df = pd.concat([distances_df, pd.DataFrame({'Name': [row.values[0]], 'a3_distance': [distance]})], axis=0, ignore_index=True)
            else:
                distances_df['d'+str(i)]= distance
    return distances_df

def get_emd_distance(vec_a, vec_b):
    return wasserstein_distance(vec_a, vec_b)




def row_emd(row, df, col_name):
    hist1 = df.loc[df['name'] == row["name1"], col_name].values[0][1]
    hist2 = df.loc[df['name'] == row["name2"], col_name].values[0][1]
    return wasserstein_distance(hist1, hist2)

def weighted_l_p_norm(vec_a, vec_b, weights, p=2):
    dists = np.abs(vec_a - vec_b)
    dists = dists * weights
    sum = np.sum(dists ** p)
    return sum ** (1 / p)

def row_euclid(row, df, descriptor_list, weights: list):
    feature_vec1 = df.loc[df["name"] == row["name1"], descriptor_list].values
    feature_vec2 = df.loc[df["name"] == row["name2"], descriptor_list].values
    dist = weighted_l_p_norm(feature_vec1, feature_vec2, weights)
    return dist

def calc_pairwise_distances(df):
    pd.options.display.max_columns = None
    big_df = pd.merge(df["name"], df["name"], how="cross")
    big_df = big_df.rename(columns={"name_x": "name1", "name_y": "name2"})
    big_df = big_df.copy()

    # compute hist similarities
    for col_name in descriptor_shape_list:
        big_df.loc[:, col_name + "_dist"] = big_df.apply(lambda row: row_emd(row, df, col_name), axis=1)
    new_cols = [col_name + "_dist" for col_name in descriptor_shape_list]
    for col in new_cols:
        big_df[col] = (big_df[col] - big_df[col].mean()) / big_df[col].std()
        big_df[col] = big_df[col] + big_df[col].min().abs()
    #for col in new_cols:
    #    print(col, big_df[col].mean(), big_df[col].std(), big_df[col].min(), big_df[col].max(), big_df[col].sum())
    big_df.loc[:, "Hist_dist"] = big_df.loc[:, new_cols].mul(hist_weights).sum(axis=1)
    big_df.drop(columns=new_cols, inplace=True)

    # compute similarity of elementary descriptors
    big_df.loc[:, "Eucl_dist"] = big_df.apply(lambda row: row_euclid(row, df, descriptor_list, elem_weights), axis=1)
    big_df.loc[:, "dist"] = big_df.loc[:, ["Hist_dist", "Eucl_dist"]].mul(common_weights).sum(axis=1)
    big_df.drop(columns=["Hist_dist", "Eucl_dist"], inplace=True)

    #print(big_df.head(20))
    #print(len(big_df))
    return big_df



def main()->None:
    database = Database()
    filename = os.path.abspath(os.path.join(current_dir, "csv_files", "aaaaaaaaaaall_desc_standardized.csv"))
    database.load_table(filename)
    df = database.get_table()
    filename = os.path.abspath(os.path.join(current_dir, "csv_files", "shape_descriptors_small_bins_standardized.csv"))
    database.clear_table()
    database.load_table(filename)
    df2 = database.get_table()
    result = pd.merge(df, df2, on='name', how='inner')
    final_result = calc_pairwise_distances(result)

    database.clear_table()
    database.add_table(final_result, "distances_test")
    database.save_table(os.path.abspath(os.path.join(current_dir, "csv_files", "distances.csv")))

    #similar_objects_eucl_df = calculate_euclidean_distances(df, "AircraftBuoyant/m1337.obj")

    #similar_objects_emd_df = get_emd(df2, "AircraftBuoyant/m1337.obj")
    #final_result = pd.merge(similar_objects_eucl_df, similar_objects_emd_df, on='Name', how='inner')
    print("FINISHED")
    # Display the result
if __name__ == "__main__":
    main()

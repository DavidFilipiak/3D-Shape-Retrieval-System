import pandas as pd

from src.query import get_kdtree


def evaluate(df):
    M_avg = 0  # overall average quality for entire database.
    M = 0  # average quality values for all class labels. M(C) = average quality value for class C

    print(f"currently processing mesh with name{df['name']}")
    dfclass_sizes = df.groupby('class_name').size()
    # This will store your results in the format: [[query_shape, retrieved_shapes], ...]
    results_list = []
    for _, row in df.iterrows():
        id = row['name']
        result = get_kdtree(id, df, dr="t-sne")
        print(f"retrieved nearest neighbors for{id}")
        # Converting the result to the desired format
        query_shape = result[0][0]
        retrieved_shapes = [item[0] for item in result[1:]]

        # Append to results_list
        results_list.append([query_shape, ", ".join(retrieved_shapes)])

    # Convert results_list to DataFrame
    print("converting results to dataframe")
    results_df = pd.DataFrame(results_list, columns=['query shape', 'retrieved shapes'])

    # Save to CSV
    results_df.to_csv('results.csv', index=False)


def compute_precision(key, matches):
    # Extract the type of the key
    query_type = key.split("/")[0]

    # Count how many matches have the same type as the query
    matching_types = sum(1 for match in matches if match[0].split("/")[0] == query_type)

    # Compute precision
    precision = matching_types / len(matches)
    return precision


# for key, matches in data.items():
#     print(f"Precision for {key}: {compute_precision(key, matches)}")
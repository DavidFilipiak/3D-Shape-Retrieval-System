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
    results_df.to_csv('evaluation_retrieval.csv', index=False)


def compute_precision(key, matches):
    # Extract the type of the key
    query_type = key.split("/")[0]

    # Count how many matches have the same type as the query
    matching_types = sum(1 for match in matches if match[0].split("/")[0] == query_type)

    # Compute precision
    precision = matching_types / len(matches)
    return precision

# def compute_accuracy(df):
#     TP = 0
#     FP = 0
#     df['retrieved shapes'] = df['retrieved shapes'].apply(lambda x: ', '.join([i.split('/')[0] for i in x.split(', ')]))
#
#     for index, row in df.iterrows():
#         query_class = row['query shape'].split('/')[0]
#
#         # List of classes for retrieved shapes
#         retrieved_classes = [shape.split('/')[0] for shape in row['retrieved shapes'].split(', ')]
#
#         # Calculate TP
#         TP += retrieved_classes.count(query_class)
#
#         # Calculate FP (needed for TN calculation)
#         FP += len(retrieved_classes) - retrieved_classes.count(query_class)
#     total_query_class = df['query shape'].str.count(query_class).sum()
#     FN = total_query_class - TP
#     # Calculate TN (assuming you have the total number of each class in the database)
#
#     # Total possible retrievals - (TP + FP + FN)
#     TN = 5 * df.shape[0] - TP - FP - FN
#     return TP, TN


def compute_metrics_for_class(df, query_class):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for index, row in df.iterrows():
        if(row['query shape'].split('/')[0] != query_class):
            continue
        # List of classes for retrieved shapes
        retrieved_classes = [shape.split('/')[0] for shape in row['retrieved shapes'].split(', ')]
        # Calculate TP
        TP += retrieved_classes.count(query_class)
        # Calculate FP
        FP += len(retrieved_classes) - retrieved_classes.count(query_class)
    # Total occurrences of the query class in the dataset
    total_query_class = df['query shape'].str.count(query_class).sum()
    # Calculate FN
    FN = (5 * total_query_class - 1) - TP
    # Calculate TN
    TN = 5 * df.shape[0] - TP - FP - FN
    return TP, FN, FP, TN

def compute_accuracy(df):
    df['retrieved shapes'] = df['retrieved shapes'].apply(lambda x: ', '.join([i.split('/')[0] for i in x.split(', ')]))

    # Extract unique query classes
    unique_query_classes = df['query shape'].apply(lambda x: x.split('/')[0]).unique()

    # Compute metrics for each unique class
    results = {}
    for query_class in unique_query_classes:
        TP, FN, FP, TN = compute_metrics_for_class(df, query_class)
        print(f"accuracy for class {query_class} is {(TP + TN) / (TP + TN + FP + FN)}")
        print(f"precision for class {query_class} is {TP / (TP + FP)}")
        print(f"recall for class {query_class} is {TP / (TP + FN)}")
        print(f"f1 for class {query_class} is {2 * TP / (2 * TP + FP + FN)}")
        results[query_class] = {
            'TP': TP,
            'FN': FN,
            'FP': FP,
            'TN': TN,
            'Accuracy': (TP + TN) / (TP + TN + FP + FN),
            'Precision': TP / (TP + FP),
            'Recall': TP / (TP + FN),
            'F1': 2 * TP / (2 * TP + FP + FN)
        }
    # Convert results dictionary to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')

    # Sort by F1 scores in descending order
    sorted_df = results_df.sort_values(by='F1', ascending=False)
    return results

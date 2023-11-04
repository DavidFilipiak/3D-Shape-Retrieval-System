import pandas as pd
from matplotlib import pyplot as plt

from src.query import get_kdtree


def evaluate(df):
    print(f"currently processing mesh with name{df['name']}")
    dfclass_sizes = df.groupby('class_name').size()
    # This will store your results in the format: [[query_shape, retrieved_shapes], ...]
    results_list = []
    for _, row in df.iterrows():
        id = row['name']
        result = get_kdtree(id, df, dr="t-sne",method = "custom")
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





def compute_metrics_for_class(df, query_class):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for index, row in df.iterrows():
        if(row['query shape'].split('/')[0] != query_class):
            continue
        total_query_class = df['query shape'].str.count(query_class).sum()
        # List of classes for retrieved shapes
        retrieved_classes = [shape.split('/')[0] for shape in row['retrieved shapes'].split(', ')]
        # Calculate TP
        TP = retrieved_classes.count(query_class)
        # Calculate FP
        FP = len(retrieved_classes) - retrieved_classes.count(query_class)

        # Calculate FN
        FN = total_query_class - TP
        # Calculate TN
        TN = df.shape[0] - 5 - FP
    # Total occurrences of the query class in the dataset


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
    plot_perclass_metrics(results_df, "Precision")
    plot_perclass_metrics(results_df, "Recall")
    plot_perclass_metrics(results_df, "F1")
    plot_perclass_metrics(results_df, "Accuracy")
    return results_df


def plot_perclass_metrics(df, title):
    # Ensure title is a valid column name
    if title not in df.columns:
        raise ValueError(f"{title} is not a column in the DataFrame")

    # Sort the DataFrame based on the title column
    df_sorted = df.sort_values(by=title)

    # Get the labels and values
    labels = df_sorted.index.tolist()
    values = df_sorted[title].tolist()

    # Plot the bar chart
    plt.figure(figsize=(10, 8))  # You can adjust the figure size to fit your needs
    plt.bar(labels, values, color='blue', width=0.7)
    plt.xlabel("Classes")
    plt.xticks(rotation=90)  # Rotate the x-axis labels so they fit and are readable
    plt.ylabel(title)
    plt.title(f"{title} per class")
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.show()



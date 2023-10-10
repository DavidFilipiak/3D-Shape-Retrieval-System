import pandas as pd
import numpy as np
import os
import ast
from feature import vector_feature_list


def string_to_np_array(string):
    try:
        #vectors
        split = string.split(" ")
        split[0] = split[0][1:]
        split[-1] = split[-1][:-1]
        return np.array([float(x) for x in split if x != ""])
    except ValueError as e:
        #histograms
        split = string.split("], [")
        split[0] = split[0][2:]
        split[-1] = split[-1][:-2]
        return np.array([[float(x) for x in y.split(", ")] for y in split])

def array_to_string(array):
    return str(array.tolist())


class Database:
    def __init__(self):
        self.table = pd.DataFrame()
        self.table_name = ""

    def load_tables(self, folder: str) -> None:
        for table in os.listdir(folder):
            new_table = self.new_table(path=os.path.join(folder, table))
            self.add_table(new_table)

    def load_table(self, path: str) -> None:
        new_table = self.new_table(path=path)
        for f in vector_feature_list:
            if f in new_table.columns:
                new_table[f] = new_table[f].apply(string_to_np_array)
        self.add_table(new_table, name=os.path.basename(path))

    def add_table(self, table: pd.DataFrame, name="") -> None:
        self.table = table
        if name != "":
            self.table_name = name

    def new_table(self, path="", df=None) -> pd.DataFrame:
        if path != "" and df is not None:
            raise Exception("You can't set path and df at the same time")
        if path != "":
            new_table = pd.read_csv(path)
        elif df is not None:
            new_table = df
        else:
            raise Exception("You must set path or df")

        return new_table

    def get_table(self) -> pd.DataFrame:
        return self.table

    def save_table(self, path: str) -> None:
        for f in vector_feature_list:
            if f in self.table.columns:
                self.table[f] = self.table[f].apply(array_to_string)
        self.table.to_csv(path, index=False)

    def clear_table(self) ->None:
        self.table = None
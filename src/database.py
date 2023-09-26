import pandas as pd
import csv_files
import os

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
        self.table.to_csv(path, index=False)
    def clear_table(self) ->None:
        self.table = None
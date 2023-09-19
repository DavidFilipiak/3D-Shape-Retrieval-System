import pandas as pd
import csv
import os

class Database:
    def __init__(self):
        self.tables = {}

    def load_tables(self, folder: str) -> None:
        for table in os.listdir(folder):
            new_table = self.new_table(path=os.path.join(folder, table))
            self.add_table(table.split(".")[0], new_table)

    def add_table(self, table_name: str, table: pd.DataFrame) -> None:
        self.tables[table_name] = table

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

    def get_table(self, table_name) -> pd.DataFrame:
        return self.tables[table_name]

    def get_table_names(self) -> list:
        return list(self.tables.keys())
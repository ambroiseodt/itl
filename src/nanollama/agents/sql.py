"""
SQL Agent

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import sqlite3
from types import TracebackType


class SQLiteDB:
    def __init__(self, db_path: str, name: str = "", columns: list[str] = None):
        """
        Initialize the SQLiteDB with the path to the database file.

        ### Parameters
        - db_path: the path to the SQLite database file
        - name: the name of the table
        - columns: a list of column names
        """
        self.db_path = db_path
        self.connection = None
        self.cursor = None

        self.columns = [] if columns is None else columns
        self.name = name

    def __enter__(self):
        """Enter SQL agent runtime context."""
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        """Exit SQL agent runtime context."""
        if self.connection:
            if exc is None:
                self.connection.commit()
            self.connection.close()

    def create(self, name: str, columns: list[str]) -> None:
        """
        initialize the database by creating a table

        ### Parameters
        - table_name: the name of the table
        - columns: a list of column names
        The first column is considered the primary key
        """
        self.columns = columns
        self.name = name
        query = f"""
            CREATE TABLE IF NOT EXISTS {name} (
                {", ".join(f"{col} TEXT" for col in columns)},
                PRIMARY KEY ({columns[0]})
            )
        """
        self.cursor.execute(query)

    def insert_element(self, person: dict[str, str]) -> None:
        """
        Insert an element into the database.

        ### Parameters
        - person: a dictionary of column names and values describing a person
        """
        keys, values = zip(*person.items())
        query = f"""
            INSERT OR REPLACE INTO {self.name} ({", ".join(keys)})
            VALUES ({", ".join(["?" for _ in keys])})
        """
        self.cursor.execute(query, values)

    def insert_elements(self, persons: list[dict[str, str]]) -> None:
        """
        Insert multiple elements into the database.

        ### Parameters
        - persons: a list of dictionaries of describing people
        """
        for person in persons:
            self.insert_element(person)

    def query(self, prompt: str) -> list[tuple[str]]:
        """Execute a query and return the results."""
        self.cursor.execute(prompt)
        return self.cursor.fetchall()

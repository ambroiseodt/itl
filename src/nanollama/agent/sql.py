# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module defining a SQL Agent

@ 2025, Meta
"""

import re
import sqlite3
from types import TracebackType

from .actor import Actor

NAME = "people"
COLUMNS = ["name", "birth_date", "birth_place", "current_address", "occupation"]


class SQLAgent:
    """
    SQL agent based on sqlite3.

    ### Parameters
    - db_path: the path to the SQLite database file
    - name: the name of the table
    - columns: a list of column names

    ### Attributes
    - pattern: the pattern to extract SQL queries `(```sql ... ```)`
    - llm_query: the format of the LLM query (`FIND ... FOR ...`)
    - sql_query: the format of the SQL query (`SELECT ... FROM ...`)
    - actor: the `source` associated with the agent
    """

    pattern = r"```sql\n(.*?)\n```"
    llm_query = r"FIND (\w+) FOR (.+)"
    sql_query = r"SELECT {attribute} FROM people WHERE name = ?"
    actor = Actor.database

    def __init__(self, db_path: str, name: str = NAME, columns: list[str] = None):
        self.db_path = db_path
        self.connection = None
        self.cursor = None

        self.columns = COLUMNS if columns is None else columns
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

    # --------------------------------------------------------------------------
    # Database creation utilities
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Agentic behavior
    # --------------------------------------------------------------------------

    def query(self, query: str, args: tuple[str]) -> list[tuple[str]]:
        """
        Execute a query and return the results.

        ### Parameters
        - query: the SQL query to execute (e.g., "SELECT * FROM table")
        - args: arguments to pass to the query (useful to avoid SQL injection)
        """
        try:
            self.cursor.execute(query, args)
        except sqlite3.OperationalError as e:
            return [(str(e),)]
        return self.cursor.fetchall()

    def execute(self, prompt: str) -> str:
        """
        Answer a LLM prompt by parsing instruction, executing them, and answering the LLM in text space.

        In our setting the instruction are not SQL query, but in a simpler language.
        This is useful to avoid SQL injection.

        ### Parameters
        - prompt: the prompt to execute

        ### Returns
        - answer: agent response
        """
        instructions = re.findall(self.pattern, prompt, re.DOTALL)
        res = []
        for instruction in instructions:
            groups = re.match(self.llm_query, instruction)
            attribute = groups.group(1)
            name = groups.group(2)
            query = self.sql_query.format(attribute=attribute)
            query_res = self.query(query, (name,))
            res.extend([row[0] for row in query_res])

        answer = ", ".join(res)
        if not answer:
            answer = "No result found"
        return answer

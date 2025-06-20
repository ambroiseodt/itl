# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Create a SQLite database and populate it with data from a JSONL file.

@ 2025, Meta
"""

import json
from pathlib import Path

from src.nanollama.agent import SQLAgent

# default variable
from .generate import ALPHA

DATA_DIR = Path().home() / f"dependent_data_{ALPHA}" / "memory"
SAVE_DIR = Path().home() / f"dependent_data_{ALPHA}" / "memory"


def create_database() -> None:
    """
    Create a SQLite database and populate it with data from a JSONL file.
    """
    with SQLAgent(SAVE_DIR / "people.db") as database:
        database.create("people", ["name", "birth_date", "birth_place", "current_address", "occupation"])

        # Read the JSONL file and insert data into the database
        with open(DATA_DIR / "people.jsonl") as file:
            for line in file:
                person = json.loads(line)
                database.insert_element(person)


def query_database(prompt: str = "SELECT * FROM people") -> None:
    """
    Query the SQLite database and print the results.

    ### Parameters
    - prompt: the SQL query to execute
    """
    with SQLAgent(SAVE_DIR / "people.db") as database:
        results = database.query(prompt)
        for row in results:
            print(row)


def delete_database() -> None:
    """
    Delete the SQLite database.
    """
    db_path = SAVE_DIR / "people.db"
    if db_path.exists():
        db_path.unlink()


if __name__ == "__main__":
    import fire

    fire.Fire({"create": create_database, "query": query_database, "delete": delete_database})

# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module defining actors

@ 2025, Ambroise Odonnat
"""

from enum import Enum


class Actor(str, Enum):
    """
    Potential interlocutor in a dialog, it could be:
    - a `user` (i.e a human) asking a question
    - an `assistant` (i.e. an LLM) answering

    It may also be tools, in particular:
    - a `database` providing response to a query
    """

    user = "user"
    assistant = "assistant"
    database = "database"

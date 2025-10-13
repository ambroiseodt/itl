# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module containing utilities to prompt an LLM with questions from Trivia QA to be formatted to allow
query from a database, e.g., using Python.

Copyright (c) 2025 by the authors
"""

import asyncio
import logging
from typing import Any

from datasets import load_dataset

# Configuration
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"
CONCURRENT_REQUESTS = 10  # Adjust based on your server capacity
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


SYSTEM_PROMPT = "You are a helpful assistant. Your task is to help a user answer a question from the Trivia QA dataset. \
                            You will be provided the question as a simple string. Once you have read the question, you will query a database of facts by outputting a special tag <query>, the question itself, and another special tag <\query>. \
                                You will then be provided with the query result, which you must then repeat to answer the question. Only repeat the answer, do not repeat the question and do not include any additional text.\n\
                                Here is an example: \n \
                                Question: What is the capital of France? \n \
                                Your first answer: <query> What is the capital of France? <\query> \n \
                                Database response: Paris \n \
                                Your final answer: Paris."


def _pick_answer(ans_obj: Any) -> tuple[str, list[str]]:
    """
    TriviaQA 'answer' can be:
      - {"value": str, "aliases": [str, ...]}
      - str
      - [str, ...]
    Returns (canonical_value, aliases_list).
    """
    if isinstance(ans_obj, dict):
        val = (ans_obj.get("value") or "").strip()
        aliases = [a.strip() for a in (ans_obj.get("aliases") or []) if isinstance(a, str) and a.strip()]
        return val, aliases
    if isinstance(ans_obj, str):
        return ans_obj.strip(), []
    if isinstance(ans_obj, list) and ans_obj:
        return (ans_obj[0] or "").strip(), [str(x).strip() for x in ans_obj[1:] if str(x).strip()]
    return "", []


def get_data(n_facts: int) -> list:
    ds = load_dataset("trivia_qa", "unfiltered", split="validation")
    qa_list = []
    for i in range(n_facts):
        line = ds[i]
        question_id = str(line.get("question_id", i))
        question = (line.get("question") or "").strip()
        ans_val, ans_aliases = _pick_answer(line.get("answer"))
        qa_list.append((question_id, question, ans_val, ans_aliases))

    return qa_list


# Call to LLM with specific prompt - WIP
from openai import AsyncOpenAI


async def database_query_async(client: AsyncOpenAI, question: str, semaphore: asyncio.Semaphore) -> dict[str, Any]:
    """
    Query database to answer a question from Trivia QA asynchronously with retry logic and rate limiting.
    """
    async with semaphore:  # Limit concurrent requests
        for attempt in range(MAX_RETRIES):
            try:
                chat_response = await client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": f"Question: {question}.\n"
                        },
                    ],
                )

                # TODO: Parse and validate the database query
                try:
                    query = int(chat_response.choices[0].message.content.strip())
                    # Design validation to raise some error

                except Exception as e:
                    print(f"Error parsing the question {question}: {e}")
                    query = "error"

                return {"question": question, "query": query}

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for paper {question}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (2**attempt))  # Exponential backoff
                else:
                    # Final attempt failed, return error result
                    return {"question": question, "query": "error"}


def database_query(n_facts: int = 500) -> None:
    qa_list = get_data(n_facts=n_facts)
    queries = []

    # TODO: define client and semaphore
    client, semaphore = None, None

    # Recover for each question-answer pairs the query to the database
    for qa_item in qa_list:
        question_id, question, ans_val, ans_aliases = qa_item
        query = database_query_async(client=client, question=question, semaphore=semaphore)
        queries.append(query)

    # TODO: do something with the query to recover the answer using Python for instance
    return


if __name__ == "__main__":
    import fire

    logging.basicConfig(level=logging.INFO)

    fire.Fire(
        {
            "query": database_query,
        }
    )

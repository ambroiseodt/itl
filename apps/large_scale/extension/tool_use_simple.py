# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module containing utilities to prompt an LLM with questions from Trivia QA to be formatted to allow
query from a database, e.g., using Python.

Copyright (c) 2025 by the authors
"""

import asyncio
import logging
from typing import Any

from openai import OpenAI

from datasets import load_dataset

# Configuration
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"
CONCURRENT_REQUESTS = 10  # Adjust based on your server capacity
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


# SYSTEM_PROMPT = """You are a helpful assistant. Your task is to help a user answer a question from the Trivia QA dataset.\n \
#                             You will be provided the question as a simple string. Once you have read the question, you will query a database of facts by outputting a special tag <query>, the question itself, and another special tag </query>. \
#                                 You will then be provided with the query result, which you must then repeat to answer the question.\n
#                                 Repeat the answer exactly as it is given to you; repeat the entirety of the answer, and do not include any additional text, explanation, rephrasing of the question, or punctuation marks.\n\
#                                 Here is an example:\n \
#                                 Question: In what state was playwright Tennessee Williams born?.\n \
#                                 <query> In what state was playwright Tennessee Williams born? </query>\n \
#                                 In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
#                                 In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
#                                 (end of the example)\n \
#                                 Here is another example:\n \
#                                     Question: Who was the only Englishman to become Pope?.\n \
#                                     <query> Who was the only Englishman to become Pope? </query>\n \
#                                     Nicholas Breakspear, who was Adrian IV from 1154 to 1159\n \
#                                     Nicholas Breakspear, who was Adrian IV from 1154 to 1159\
#


SYSTEM_PROMPT = """You are a helpful assistant. Your task is to help a user answer a question from the Trivia QA dataset.\n \
You will be provided the question as a simple string.
If you only have access to the question, then you must output a special tag <query>, the question itself, and another special tag </query>. \
Here is an example: \n\
Question: In what state was playwright Tennessee Williams born?.\n \
(your answer) <query> In what state was playwright Tennessee Williams born? </query>\n \
You MUST Output exactly "<query> the question </query>" \n\
You MUST not output anything else at this point.\n\
First example:
Question: In what state was playwright Tennessee Williams born?.\n \
<query> In what state was playwright Tennessee Williams born? </query>\n \
Second example:
Question: Who was the only Englishman to become Pope?.\n \
<query> Who was the only Englishman to become Pope? </query>\n \
Outputting <query> question </query> will automatically make a database query, and the answer of that query will be output in the chat.\n \
If you are given the question, if you have already output the <query> tags with the question repeated, and if you have been provided with an answer from the database, then you must repeat that answer to answer the question.\n\
You MUST repeat the answer exactly as it is given to you. Repeat the entirety of the answer. Do not include any additional text, explanation, rephrasing of the question, or punctuation marks.\n\
Here is a first, including the early steps of the conversation:\n \
Question: In what state was playwright Tennessee Williams born?.\n \
<query> In what state was playwright Tennessee Williams born? </query>\n \
In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
Here is a second example:\n \
Question: Who was the only Englishman to become Pope?.\n \
<query> Who was the only Englishman to become Pope? </query>\n \
Nicholas Breakspear, who was Adrian IV from 1154 to 1159\n \
Nicholas Breakspear, who was Adrian IV from 1154 to 1159\n \
REMEMBER: if you are given only a question, you must output the <query> tags with the question repeated between them, and nothing else. \
If you are given a question, if you have already output the <query> tags with the question repeated, and if you have been provided with an answer from the database, then you must repeat the answer, and nothing else.\
"""


# SYSTEM_PROMPT = """You are a helpful assistant. Your task is to help a user answer a question from the Trivia QA dataset.\n \
# You will be provided the question as a simple string.
# Once you have read the question, you must output a special tag <query>, the question itself, and another special tag </query>. \
# Here is an example: \n\
# Question: In what state was playwright Tennessee Williams born?.\n \
# (your answer) <query> In what state was playwright Tennessee Williams born? </query>\n \
# You MUST Output exactly "<query> the question </query>" \n\
# You MUST not output anything else at this point.\n\
# You will then be provided with an answer, which you must then repeat to answer the question.\n\
# Here is an example:\n\
# (the answer that you are given) In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
# (your final answer) In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
# You MUST repeat the answer exactly as it is given to you. Repeat the entirety of the answer. Do not include any additional text, explanation, rephrasing of the question, or punctuation marks.\n\
# Here is an example, including the early steps of the conversation:\n \
# Question: In what state was playwright Tennessee Williams born?.\n \
# (your first answer) <query> In what state was playwright Tennessee Williams born? </query>\n \
# (the answer that you are given) In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
# (your final answer) In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
# Without the indications in parentheses, this is the format you should follow:\n \
# Question: In what state was playwright Tennessee Williams born?.\n \
# <query> In what state was playwright Tennessee Williams born? </query>\n \
# In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
# In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
# Here is another example:\n \
# Question: Who was the only Englishman to become Pope?.\n \
# <query> Who was the only Englishman to become Pope? </query>\n \
# Nicholas Breakspear, who was Adrian IV from 1154 to 1159\n \
# Nicholas Breakspear, who was Adrian IV from 1154 to 1159\
"""


# Question: In what state was playwright Tennessee Williams born?.\n \
#                                 Your first answer: <query> In what state was playwright Tennessee Williams born? </query>\n \
#                                 Database response: In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state\n \
#                                 Your final answer: In Mississippi, as Thomas Lanier Williams. He took the name Tennessee after his father's home state"""








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


# # Call to LLM with specific prompt - WIP
# from openai import AsyncOpenAI



def query_model(client: OpenAI, question: str, answer: str) -> bool:
    """
    Have the model answer a question from Trivia QA. Returns True if the answer is correct, False otherwise.
    """
    chat_response = client.chat.completions.create(
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
    # Check if the query has the structure <query> question <\query>, where question is the input question
    query = chat_response.choices[0].message.content.strip()
    if query is not None and query.startswith("<query>") and query.endswith("</query>"):
        extracted_question = query[len("<query>") : -len("</query>")].strip()
        if extracted_question != question:
            return False
    chat_response = client.chat.completions.create(
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
            {
                "role": "assistant",
                "content": query + "\n"
            },

            {
                "role": "system",
                "content": answer + "\n"
            },
        ],
    )
    final_answer = chat_response.choices[0].message.content.strip()

    print("----")
    print(f"Question: {question}.\n")
    print(query + "\n")
    print(answer + "\n")
    print(final_answer + "\n")
    if final_answer.lower() == answer.lower():
        return True
    else:
        print("False !")
        print(f"Final answer was: {final_answer.lower()}")
        print(f"Expected answer was: {answer.lower()}")
    print("----")
    return False




if __name__ == "__main__":

    N = 1000

    data = get_data(n_facts=N)


    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )
    num_correct = 0
    total = 0
    for data_item in data:
        question_id, question, ans_val, ans_aliases = data_item
        correct = query_model(client=client, question=question, answer=ans_val)
        if correct:
            num_correct += 1
        total += 1
    print(f"Final accuracy: {num_correct}/{total} = {num_correct/total:.2%}")


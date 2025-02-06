"""
Generate a list of entities with random attributes.

License
-------
This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

@ 2025, the authors
"""

import json
import random
from itertools import product
from pathlib import Path

from jinja2 import Template

seed = 42
SAVE_DIR = Path(__file__).resolve().parent


def generate_people() -> None:
    """
    Generate a list of people with random attributes.
    """
    global first_names, last_names, cities, countries, occupations
    keys = ["first_names", "last_names", "cities", "countries", "occupations"]
    for key in keys:
        with open(SAVE_DIR / "atoms" / f"{key}.txt") as f:
            globals()[key] = f.read().splitlines()

    # shuffle people ordering
    random.seed(seed)
    all_pairs = list(product(first_names, last_names))
    random.shuffle(all_pairs)

    # generate all unique combinations of first and last names and save to file
    save_file = SAVE_DIR / "people.jsonl"
    with open(save_file, "w") as f:
        for first_name, last_name in all_pairs:
            entity = {
                "name": f"{first_name} {last_name}",
                "birth_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/{random.randint(1950, 2000)}",
                "birth_place": f"{random.choice(countries)}",
                "current_address": f"{random.choice(cities)}",
                "occupation": random.choice(occupations),
            }
            print(json.dumps(entity), file=f, flush=True)


def collect_people(num: int = float("inf")) -> list[dict[str, str]]:
    """
    Collect a list of people with random attributes.

    Parameters
    ----------
    num: Number of people to collect (by default, it collect everyone).
    """
    save_file = SAVE_DIR / "people.jsonl"
    people = []
    with open(save_file) as f:
        while len(people) < num:
            line = f.readline()
            if not line:
                break
            people.append(json.loads(line))

    return people


def generate_biographies(num: int = float("inf")) -> None:
    """
    Generate a list of biographies of people.
    """
    # recover templates
    templates: list[Template] = []
    for file in (SAVE_DIR / "templates").glob("bio*.j2"):
        with open(file) as f:
            templates.append(Template(f.read()))

    # write biographies to file
    with open(SAVE_DIR / "biographies.jsonl", "w") as f:
        for i, people in enumerate(collect_people(num=num)):
            for template in templates:
                biography = template.render(**people)
                print(json.dumps({"text": biography, "people_id": i}), file=f, flush=True)


def generate_qa(num: int = float("inf"), tooluse: bool = False) -> None:
    """
    Generate a list of questions about people biographies.

    Notes
    -----
    This is a simple implementation, a better implementation would store the attribute to retrieve.
    It should also make sure that the `<TOOLUSE>` tag will be tokenized as a special token.
    """
    templates: list[Template] = []
    for file in (SAVE_DIR / "templates").glob("qa*.j2"):
        with open(file) as f:
            templates.append(Template(f.read()))

    render = {}
    if tooluse:
        render["tooluse"] = "<TOOLUSE> request from database </TOOLUSE> "
    else:
        render["tooluse"] = ""

    with open(SAVE_DIR / "qa.jsonl", "w") as f:
        for i, people in enumerate(collect_people(num=num)):
            for template in templates:
                qa = template.render(**people | render)
                print(json.dumps({"text": qa, "people_id": i}), file=f, flush=True)


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "people": generate_people,
            "biographies": generate_biographies,
            "qa": generate_qa,
        }
    )

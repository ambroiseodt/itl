# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Data utilities

@ 2025, Ambroise Odonnat
"""

from numpy.random import SeedSequence

# ------------------------------------------------------------------------------
# Randomness utils
# ------------------------------------------------------------------------------


def generate_seeds(
    nb_shared: int, nb_individual: int, root_seed: int, rank: int
) -> tuple[list[SeedSequence], list[SeedSequence]]:
    """
    Generate seeds for various workers

    Parameters
    ----------
    - nb_shared: number of seeds shared across workers
    - nb_individual: number of seeds specific to each workers
    - root_seed: initial seed to spawn new seeds
    - rank: worker rank
    - world_size: total number of worker
    """
    nb_seeds = nb_shared + nb_individual * (rank + 1)
    seeds = SeedSequence(root_seed).spawn(nb_seeds)
    shared_seeds = seeds[:nb_shared]
    individual_seeds = seeds[-nb_individual:]
    return shared_seeds, individual_seeds

"""
Data utilities

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from numpy.random import SeedSequence

# ------------------------------------------------------------------------------
# Randomness utils
# ------------------------------------------------------------------------------


def generate_seeds(nb_shared: int, nb_individual: int, root_seed: int, rank: int, world_size: int) -> tuple[list, list]:
    """
    Generate seeds for various workers

    Parameters
    ----------
    nb_shared: number of seeds shared across workers
    nb_individual: number of seeds specific to each workers
    root_seed: initial seed to spawn new seeds
    rank: worker rank
    world_size: total number of worker
    """
    nb_seeds = nb_shared + nb_individual * (rank + 1)
    seeds = SeedSequence(root_seed).spawn(nb_seeds)
    shared_seeds = seeds[:nb_shared]
    individual_seeds = seeds[-nb_individual:]
    return shared_seeds, individual_seeds


if __name__ == "__main__":

    def test_seeds() -> None:
        nb_shared = 10
        nb_individual = 5
        root_seed = 42
        world_size = 16
        base_ss = None
        base_is = None
        for rank in range(world_size):
            s_s, i_s = generate_seeds(nb_shared, nb_individual, root_seed, rank, world_size)
            assert len(s_s) == nb_shared
            if base_ss is None:
                base_ss = s_s
            else:
                for base, seed in zip(base_ss, s_s):
                    assert base.entropy == seed.entropy
                    assert base.spawn_key == seed.spawn_key
            assert len(i_s) == nb_individual
            if base_is is None:
                base_is = i_s
            else:
                for base, seed in zip(base_is, i_s):
                    assert base.entropy == seed.entropy
                    assert base.spawn_key != seed.spawn_key

    test_seeds()

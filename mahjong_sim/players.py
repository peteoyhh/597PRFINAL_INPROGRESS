from typing import Union
import random


class NeutralPolicy:
    """
    Soft-defensive neutral player policy used for Experiment 1 tables.

    Rules:
        - If risk > 0.4 -> Hu immediately (avoid danger)
        - If fan >= 1 -> Hu
        - Otherwise 20% chance to continue chasing higher fan
    """

    def __init__(self, seed: Union[int, None] = None) -> None:
        self._rng = random.Random(seed)

    def should_hu(self, fan: Union[int, float], risk: float) -> bool:
        if risk > 0.4:
            return True
        if fan >= 1:
            return True
        # 20% chance to chase higher fan (i.e., continue drawing)
        return self._rng.random() >= 0.2


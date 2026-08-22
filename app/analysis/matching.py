from __future__ import annotations

import math
from collections.abc import Iterable


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    if not av or len(av) != len(bv):
        return -1.0
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av))
    nb = math.sqrt(sum(y * y for y in bv))
    if na <= 1e-12 or nb <= 1e-12:
        return -1.0
    return dot / (na * nb)

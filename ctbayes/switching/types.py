from typing import NamedTuple

import numpy as np
from ctbayes.mjplib import skeleton


class Anchorage(NamedTuple):
    t: np.ndarray
    yt: np.ndarray
    vt: np.ndarray


def prune_anchorage(h: Anchorage) -> skeleton.Skeleton:

    is_self_trans = np.hstack([False, np.diff(h.yt) == 0])
    return skeleton.Skeleton(h.t[~is_self_trans], h.yt[~is_self_trans], h.t[-1])

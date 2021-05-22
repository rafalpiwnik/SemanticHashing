from __future__ import annotations

import numpy as np
from bitarray import frozenbitarray


class MedianHash:
    def __init__(self, z_mean: np.ndarray):
        """Creates a median hash from numpy ndarray"""
        if z_mean.ndim > 1:
            raise ValueError(f"Passed latent ndarray is not a vector, shape={z_mean.shape}")
        self._num_bits = len(z_mean)

        median = np.median(z_mean)
        one_hot = [str(int(x > median)) for x in z_mean]
        str_code = "".join(one_hot)

        self._code = frozenbitarray(str_code)

    @property
    def code(self) -> frozenbitarray:
        """Returns the frozenbitarray representation """
        return self.code

    def hamming(self, other: MedianHash) -> int:
        """Calculates the hamming dist between two median hash codes"""
        return (self.code ^ other.code).count()

    def __len__(self):
        return len(self.code)

    def __repr__(self):
        return self._code.__repr__()

    def __str__(self):
        return self._code.__str__()

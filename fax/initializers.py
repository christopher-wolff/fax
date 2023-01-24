import abc
from typing import Optional, Sequence

import jax
from jax import numpy as jnp


class Initializer(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        *,
        key: Optional[jax.random.PRNGKey] = None,
        **kwargs,
    ) -> jnp.array:
        pass


class ZerosInitializer(Initializer):
    def __init__(self, shape: Sequence[int]):
        self.shape = shape

    def __call__(self, key=None):
        del key
        return jnp.zeros(self.shape)


class HeInitializer(Initializer):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, key: jax.random.PRNGKey):
        maxval = -1 / self.in_features**0.5
        minval = -maxval
        return jax.random.uniform(
            key,
            shape=(self.in_features, self.out_features),
            minval=minval,
            maxval=maxval,
        )

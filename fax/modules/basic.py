"""A collection of basic modules.
"""

from typing import Optional

from jax import numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array, Float

from .core import Module
from .. import initializers


class Linear(Module):
    """A linear function, computing Ax + b.

    Initialized with uniform random weights and zero bias.
    """

    NODES = ["weight", "bias"]

    def __init__(
        self,
        key: KeyArray,
        in_features: int,
        out_features: int,
        with_bias: bool = True,
        w_init: Optional[initializers.Initializer] = None,
        b_init: Optional[initializers.Initializer] = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.with_bias = with_bias

        if w_init is None:
            w_init = initializers.HeInitializer(in_features, out_features)
        if with_bias and b_init is None:
            b_init = initializers.ZerosInitializer(shape=(out_features,))

        self.weight = w_init(key)
        if with_bias:
            self.bias = b_init(key)

    def __call__(
        self,
        inputs: Float[Array, "batch in"],
    ) -> Float[Array, "batch out"]:
        out = jnp.dot(inputs, self.weight)

        if self.with_bias:
            bias = jnp.broadcast_to(self.bias, out.shape)
            out = out + bias

        return out


class ElementwiseLinear(Module):
    """
    An elementwise linear layer.

    Initialized with unit gain and zero bias.
    """

    NODES = ["gain", "bias"]

    def __init__(self, shape):
        self.gain = jnp.ones(shape)
        self.bias = jnp.zeros(shape)

    def __call__(self, x: Float[Array, "batch in"]) -> Float[Array, "batch in"]:
        gain = jnp.broadcast_to(self.gain, x.shape)
        out = gain * x
        bias = jnp.broadcast_to(self.bias, out.shape)
        out += bias
        return out

"""A simple Transformer implementation.

Based on https://github.com/awf/functional-transformer by Andrew Fitzgibbon.
"""

import chex
import jax
from jax import numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array, Bool, Float, UInt

from .config import TransformerConfig
from ..basic import ElementwiseLinear
from ..basic import Linear
from ..core import Module


def _layer_norm(
    x: Float[Array, "d1 d2 d3"],
    eps: float = 1e-5,
) -> Float[Array, "d1 d2 d3"]:
    mean = x.mean(-1)
    std = x.std(-1)
    return (x - mean[:, None]) / (std[:, None] + eps)


class TransformerHead(Module):

    NODES = ["query", "key", "value"]

    def __init__(
        self,
        key: KeyArray,
        d_model: int,
        d_head: int,
    ):
        qkey, kkey, vkey = jax.random.split(key, num=3)
        self.query = Linear(
            key=qkey,
            in_features=d_model,
            out_features=d_head,
        )
        self.key = Linear(
            key=kkey,
            in_features=d_model,
            out_features=d_head,
        )
        self.value = Linear(
            key=vkey,
            in_features=d_model,
            out_features=d_model,
        )


class TransformerLayer(Module):

    NODES = [
        "norm_self_attn",
        "heads",
        "norm_ff",
        "ffn1",
        "ffn2",
    ]

    def __init__(
        self,
        key: KeyArray,
        n_heads: int,
        d_model: int,
        d_head: int,
        d_ff: int,
        temperature: float,
    ):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.d_ff = d_ff
        self.temperature = temperature

        self.elemwise_linear_1 = ElementwiseLinear(d_model)

        key, *subkeys = jax.random.split(key, num=n_heads + 1)
        self.heads = [
            TransformerHead(key=subkey, d_model=d_model, d_head=d_head)
            for subkey in subkeys
        ]

        self.elemwise_linear_2 = ElementwiseLinear(d_model)

        subkey1, subkey2 = jax.random.split(key, num=2)
        self.ff1 = Linear(subkey1, d_model, d_ff)
        self.ff2 = Linear(subkey2, d_ff, d_model)

    def __call__(
        self,
        x: Float[Array, "batch seq dm"],
        mask: Bool[Array, "batch seq"],
    ) -> Float[Array, "batch seq dm"]:
        bsz, seqlen, _ = x.shape
        dh, dm = self.d_head, self.d_model

        # Layer-normalize.
        t1 = jax.vmap(_layer_norm)(x)
        t1 = self.elemwise_linear_1(t1)
        chex.assert_shape(t1, (bsz, seqlen, dm))

        # Multi-head self-attention.
        for head in self.heads:
            # Project into this head's query/key space.
            query = head.query(t1)
            key = head.key(t1)
            chex.assert_shape([query, key], (bsz, seqlen, dh))

            # Compute attention matrix.
            score = query @ key.swapaxes(1, 2) + mask
            chex.assert_shape(score, (bsz, seqlen, seqlen))

            attn = jax.nn.softmax(self.temperature * score, axis=1)
            chex.assert_shape(attn, (bsz, seqlen, seqlen))

            value = head.value(t1)
            chex.assert_shape(value, (bsz, seqlen, dm))
            self_attn = attn @ value
            chex.assert_shape(self_attn, (bsz, seqlen, dm))

            # Add this head's contribution.
            x += self_attn
            chex.assert_shape(x, (bsz, seqlen, dm))

        # Layer-normalize.
        t2 = jax.vmap(_layer_norm)(x)
        t2 = self.elemwise_linear_2(t2)
        chex.assert_shape(t2, (bsz, seqlen, dm))

        # Feedforward fully connected.
        t2 = self.ff1(t2)
        chex.assert_shape(t2, (bsz, seqlen, self.d_ff))
        t2 = jax.nn.relu(t2)
        t2 = self.ff2(t2)
        chex.assert_shape(t2, (bsz, seqlen, dm))

        # Residual connection.
        x += t2

        return x


class Transformer(Module):

    NODES = [
        "embeddings",
        "positional_encodings",
        "layers",
        "pre_output_norm",
        "output",
    ]

    def __init__(self, key: KeyArray, cfg: TransformerConfig):
        self.cfg = cfg

        key, subkey = jax.random.split(key)
        self.embeddings = jax.random.normal(subkey, shape=(cfg.vocab_size, cfg.d_model))
        self.positional_encodings = jnp.zeros((cfg.max_len, cfg.d_model))

        key, *subkeys = jax.random.split(key, num=cfg.n_layers + 1)
        self.layers = [
            TransformerLayer(
                key=subkey,
                n_heads=cfg.n_heads,
                d_model=cfg.d_model,
                d_head=cfg.d_head,
                d_ff=cfg.d_ff,
                temperature=cfg.temperature,
            )
            for subkey in subkeys
        ]

        self.elemwise_linear = ElementwiseLinear(cfg.d_model)
        self.output = Linear(key, cfg.d_model, cfg.vocab_size)

    def __call__(self, x: UInt[Array, "batch seq"]) -> Float[Array, "batch seq out"]:
        bsz, seqlen = x.shape
        dm = self.cfg.d_model

        # Mask entries: 0 to attend, -Inf to ignore.
        mask = jnp.log(jnp.tril(jnp.ones((seqlen, seqlen))))
        chex.assert_shape(mask, (seqlen, seqlen))

        # Lookup from embedding table.
        embeddings = self.embeddings[x, :]
        chex.assert_shape(embeddings, (bsz, seqlen, dm))

        # Add trainable positional encodings.
        pe = self.positional_encodings[:seqlen, :]
        pe = jnp.broadcast_to(pe[None, :, :], embeddings.shape)
        latents = embeddings + pe
        chex.assert_shape(latents, (bsz, seqlen, dm))

        # Apply the transformer layers.
        for layer in self.layers:
            latents += layer(latents, mask)
            chex.assert_shape(latents, (bsz, seqlen, dm))

        # Layer-normalize.
        latents = jax.vmap(_layer_norm)(latents)
        latents = self.elemwise_linear(latents)
        chex.assert_shape(latents, (bsz, seqlen, dm))

        # Linearly project to output dimension.
        output = self.output(latents)
        chex.assert_shape(output, (bsz, seqlen, self.cfg.vocab_size))
        return output

    def generate(
        self, seq: UInt[Array, "batch seq"], length: int = 20
    ) -> UInt[Array, "batch out"]:
        for _ in range(length):
            output = self(seq)
            idx = jnp.argmax(output[-1])
            seq = jnp.append(seq, idx)
        return seq

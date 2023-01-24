import dataclasses


@dataclasses.dataclass
class TransformerConfig:

    d_model: int
    d_head: int
    d_ff: int
    n_heads: int
    vocab_size: int
    n_layers: int
    max_len: int = 4096
    temperature: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.temperature = 1 / self.d_head**0.5

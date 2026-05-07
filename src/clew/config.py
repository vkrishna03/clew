from dataclasses import dataclass, field


@dataclass
class Config:
    # data
    data_dir: str = "data"
    block_size: int = 256

    # model
    vocab_size: int = 65          # set by data prep
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.0

    # ARIADNE
    lsh_bits: int = 32
    asc_k_max: int = 64
    asc_n_probes: int = 4
    aux_router_weight: float = 0.1
    aux_router_until_step: int = 500

    # training
    batch_size: int = 16
    max_steps: int = 1000
    eval_interval: int = 200
    eval_iters: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # runtime
    device: str = "mps"
    dtype: str = "float32"
    seed: int = 1337
    out_dir: str = "runs"

    @staticmethod
    def small():
        return Config()

    @staticmethod
    def tiny():
        return Config(n_layer=2, n_head=2, d_model=64, max_steps=500)

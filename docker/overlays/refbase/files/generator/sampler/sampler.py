from abc import abstractmethod
import torch
from ...tokenizer import Tokenizer

class Sampler:
    def __init__(self):
        self.reqs_past_ids = False
        self.reqs_torch_seed = False
        # Non-None only for a sampler whose result is independent for every logit row and has
        # no mutable sampling state. Generator.iterate_gen uses the key to batch MTP verification
        # only when every job has the same implementation.
        self.batch_verify_key = None

    @abstractmethod
    def forward(
        self,
        logits,
        sequence_ids: torch.Tensor | None = None,
        rand_u32: int | None = None,
        tokenizer: Tokenizer | None = None,
        blocked_tokens: list[int] | None = None,
        allowed_tokens: list[int] | None = None,
        return_state: bool = False
    ):
        pass

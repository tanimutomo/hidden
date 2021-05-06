from dataclasses import dataclass
import random
import typing

import numpy as np
import torch
import torchtext


@dataclass
class Pair:
    idx: typing.List[int]
    vec: torch.Tensor


@dataclass
class GloVe:
    use_words: int
    num_words: int
    name: str ="6B"
    dim: int =50
    
    def __post_init__(self):
        glove = torchtext.vocab.GloVe(name=self.name, dim=self.dim)
        if self.use_words > len(glove.itos):
            raise TypeError("cannot use words more than base vector")
        idxs = random.sample(range(0, len(glove.itos)), self.use_words)
        self._key = np.array(glove.itos)[idxs]
        self._vec = glove.vectors[idxs]

    def get_with_random(self) -> Pair:
        idxs = random.sample(range(0, self._vec.shape[0]), self.num_words)
        return Pair(idx=idxs, vec=self._vec[idxs])

    def most_similar(self, x: torch.FloatTensor) -> Pair:
        if x.shape[-1] != self.dim:
            raise TypeError
        vec = self._vec.to(x.device)
        idx = torch.argmin(torch.norm(vec - x.unsqueeze(-2), dim=-1), dim=-1)
        return Pair(idx=idx, vec=self._vec[idx])
    
    def serialize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.ndim < 2 or x.shape[-1] != self.dim:
            raise TypeError
        return x.view(*x.shape[:-2], -1)

    def unserialize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.shape[-1] % self.dim != 0 and (x.shape[-1] / self.dim) == self.num_words:
            raise TypeError
        return x.view(*x.shape[:-1], self.num_words, self.dim)

    def get_keys(self, idxs: typing.List[int]) -> typing.List[str]:
        return self._key[idxs].tolist()


@dataclass
class WordVector:
    idx: torch.Tensor
    vec: torch.Tensor

    def serialized(self):
        return self.vec.view(*self.vec.shape[:-2], -1)

    def to(self, device: torch.device):
        self.vec = self.vec.to(device)
        return self


from dataclasses import dataclass

import torch
import torchtext


@dataclass
class WordVector:
    idx torch.Tensor
    vec torch.FloatTensor


@dataclass
class GloVe:
    name: str ="4B"
    dim: int =50
    use_words: int
    
    def __post_init__(self):
        glove = torchtext.vocab.GloVe(name="4B", dim=50)
        if use_words > len(glove.itos):
            raise TypeError("cannot use words more than base vector")
        idxs = torch.randint(0, len(glove.itos), use_words)
        self._key = glove.itos[idxs]
        self._vec = glove.vectors[idxs]

    def to(self, device: torch.device):
        self._vec.to(device)

    def get_with_random(self, num_words: int) -> WordVector:
        idxs = torch.randint(0, self._vec.shape[0], num_words)
        return WordVector(idx=idxs, vec=self._vec[idxs])

    def most_similar(self, x: torch.FloatTensor) -> WordVector:
        if x.shape[-1] != self.dim:
            raise TypeError
        idx = torch.argmin(torch.norm(self._vec - x.unsqueeze(-2), dim=-1), dim=-1)]
        return WordVector(idx=idx, vec=self._vec[idx])
    
    def serialize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.ndim < 2 or x.shape[-1] != dim:
            raise TypeError
        return x.view(*x.shape[:-2], -1)

    def unserialize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.shape[-1] % self.dim != 0 and x.shape[-1] / self.dim == self.use_words:
            raise TypeError
        return x.view(*x.shape[:-1], self.use_words, self.dim)

    def get_keys(idxs: typing.List[int]) -> typing.List[str]:
        return self._key[idxs]

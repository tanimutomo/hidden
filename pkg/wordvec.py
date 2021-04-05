from dataclasses import dataclass

import torch
import torchtext


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
        self.key = glove.itos[idxs]
        self.vec = glove.vectors[idxs]

    def to(self, device: torch.device):
        self.vec.to(device)

    def get_with_random(self, num_words: int) -> torch.FloatTensor:
        idxs = torch.randint(0, self.vec.shape[0], num_words)
        return self.vec[idxs]

    def most_similar(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.shape[-1] != self.dim:
            raise TypeError
        return self.vec[torch.argmin(torch.norm(self.vec - x.unsqueeze(-2), dim=-1), dim=-1)]
    
    def serialize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.ndim < 2 or x.shape[-1] != dim:
            raise TypeError
        return x.view(*x.shape[:-2], -1)


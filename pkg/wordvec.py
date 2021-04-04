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
            raise ValueError("cannot use words more than base vector")
        idxs = torch.randint(0, len(glove.itos), use_words)
        self.key = glove.itos[idxs]
        self.vec = glove.vectors[idxs]

    def to(device: torch.device):
        self.vec.to(device)

    def get_with_random(num_words: int) -> torch.FloatTensor:
        idxs = torch.randint(0, self.vec.shape[0], num_words)
        return self.vec[idxs]

    def most_similar(x: torch.FloatTensor):
        if len(xs.shape) != 1:
            raise ValueError
        return self.vec[torch.argmin(torch.norm(self.vec - x, dim=1))]

    def most_similars(xs: torch.FloatTensor):
        if len(xs.shape) != 2:
            raise ValueError
        vs = []
        for x in xs:
            vs.append(self.most_similar(x))
        return torch.stack(vs, dim=0)
    

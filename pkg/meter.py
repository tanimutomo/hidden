import typing


class MultiAverageMeter(object):
    def __init__(self, names: typing.List[str]):
        self.names = names
        for name in names:
            setattr(self, name, AverageMeter())

    def updates(self, losses: dict):
        for name, value in losses.items():
            self.update(name, value)
    
    def update(self, name: str, value: float):
        loss = getattr(self, name)
        setattr(self, name, loss.update(value))

    def to_dict(self) -> dict:
        d = dict()
        for name in self.names:
            d[name] = getattr(self, name).avg


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

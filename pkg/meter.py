import typing


class MultiAverageMeter(object):
    def __init__(self, names: typing.List[str]):
        self.names = names
        for name in names:
            setattr(self, name, AverageMeter())

    def updates(self, values: dict):
        for name, value in values.items():
            self.update(name, value)
    
    def update(self, name: str, value: float):
        meter = getattr(self, name)
        meter.update(value)
        setattr(self, name, meter)

    def to_dict(self) -> dict:
        d = dict()
        for name in self.names:
            d[name] = getattr(self, name).avg
        return d


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

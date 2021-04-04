
def ptensor(name, tensor):
    if tensor.ndim == 4:
        channel_dim = 1
    elif tensor.ndim == 3:
        channel_dim = 0
    else:
        raise ValueError("cannot handle tensor which doesn't have 3 or 4 dimension")
    for c in range(tensor.shape[channel_dim]):
        if channel_dim == 1:
            t = tensor[:, c, ...]
        else:
            t = tensor[c, ...]
        print(f"{name}[{c}]: min={t.min().item():.2f} max={t.max().item():.2f} mean={t.mean().item():.2f}")


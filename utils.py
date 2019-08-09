import torch

class DeNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float).view(-1, 1, 1)

    def __call__(self, tensor):
        out = tensor * self.std + self.mean
        return out

class VideoTransform:
    def __init__(self, transform):
        self.transform = transform


    def __call__(self, video):
        return torch.stack(list(map(self.transform or identitymap, video)))


    def __bool__(self):
        return bool(self.transform)

def rearaange_temporal_batch(data_batch, T):
    B = data_batch.size(0) // T
    data_batch = data_batch.view(B, T, *data_batch.shape[1:])
    data_batch = data_batch.transpose(1, 2).contiguous()
    return data_batch.detach()

def identitymap(x):
    return x

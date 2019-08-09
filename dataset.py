import torch
import torch.utils.data
from torchvision import datasets, transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from utils import *

import numpy as np
import subprocess

class StochasticMovingMNIST(torch.utils.data.Dataset):
    def __init__(self, train, data_root='/home/chaehuny/data/dataset/mnist/', seq_len=20, num_digits=2, img_size=64, deterministic=False):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.img_size = img_size
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False
        self.channels = 1

        self.data = datasets.MNIST(
            path, train=train, download=True,
            transform=T.Compose([T.Scale(self.digit_size),
                                    T.ToTensor()])
        )

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        img_size = self.img_size
        digit_size = self.digit_size
        # x = np.zeros((self.seq_len, img_size, img_size, self.channels))
        x = torch.zeros((self.channels, self.seq_len, img_size, img_size))
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(img_size - digit_size)
            sy = np.random.randint(img_size - digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= img_size - 32:
                    sy = img_size - 32 - 1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dy = np.random.randint(-4, 5)
                        dx = np.random.randint(1, 5)
                elif sx >= img_size - 32:
                    sx = img_size - 32 - 1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dy = np.random.randint(-4, 5)
                        dx = np.random.randint(-4, 0)

                # x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
                x[:, t, sy:sy+32, sx:sx+32] += digit
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x

class MovingMNIST(torch.utils.data.Dataset):
    seqlen = 20
    normalize = T.Normalize(mean=(0.049270592390472503,),
                            std=(0.2002874575763297,))
    denormalize = DeNormalize(mean=(0.049270592390472503,),
                            std=(0.2002874575763297,))

    def __init__(self, data_path, transform, train=True, train_ratio=0.8):
        datafile = data_path  # /home/chaehuny/chaehun/dataset/MOVING_MNIST/mnist_test_seq.npy
        data = np.load(datafile)  # T, N, H, W
        self.videos = np.transpose(data, (1, 0, 2, 3))  # N, T, H, W
        N = self.videos.shape[0]
        if train:
            self.videos = self.videos[:int(N*train_ratio)]
        else:
            self.videos = self.videos[int(N*train_ratio):]

        self.transform = VideoTransform(transform)


    def __len__(self):
        return len(self.videos)


    def __getitem__(self, idx):
        v = np.expand_dims(self.videos[idx], axis=-1)
        if self.transform:
            v = self.transform(v)
        return v.transpose(1, 0)

def grid2gif(img_str, output_gif, delay=100):
    str1 = 'convert -delay ' + str(delay) + ' -loop 0 ' + img_str + ' ' + output_gif
    subprocess.call(str1, shell=True)

if __name__=='__main__':
    dataset = StochasticMovingMNIST(True, num_digits=2)
    dataloader = DataLoader(dataset, 64, shuffle=True, pin_memory=True)

    for i, video in enumerate(dataloader):
        for t in range(dataset.seq_len):
            save_image(video[:, :, t], '/home/chaehuny/data/svp/s_mnist_result/image_%03d.jpg'%(t), nrow=8)

        if i == 0:
            break


    grid2gif('/home/chaehuny/data/svp/s_mnist_result/image_*', '/home/chaehuny/data/svp/s_mnist_result/image.gif', delay=10)



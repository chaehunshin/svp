from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T

import subprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from tensorboardX import SummaryWriter

from model import SVP
from Options import TrainOptions
from dataset import MovingMNIST
from loss import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def grid2gif(img_str, output_gif, delay=100):
    str1 = 'convert -delay ' + str(delay) + ' -loop 0 ' + img_str + ' ' + output_gif
    subprocess.call(str1, shell=True)

def main():
    args = TrainOptions()
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # transforms = T.Compose([T.ToTensor(),
    #                         MovingMNIST.normalize])
    transforms = T.ToTensor()

    dataset = MovingMNIST(args.data_dir, transforms, False)
    dataloader = DataLoader(dataset, 32, True, pin_memory=True)

    result_dir = '/home/chaehuny/data/svp/result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    model = SVP(args.img_dim, args.h_dim, args.g_dim, args.latent_dim, args.rnn_size,
                args.pre_n_layers, args.pos_n_layers, args.pri_n_layers).to(device)
    _ = model.load(os.path.join('/home/chaehuny/data/svp/ckpt', args.resume))

    for i, video in enumerate(dataloader):
        video = video.to(device)
        prediction_frames = video[:, :, 0]
        # prediction_frames = MovingMNIST.denormalize(make_grid(prediction_frames, nrow=4).cpu().detach())
        prediction_frames = make_grid(prediction_frames, nrow=8).cpu().detach()
        save_image(prediction_frames, os.path.join(result_dir, 'reconstruction_image_%03d.jpg'%(0)))
        save_image(prediction_frames, os.path.join(result_dir, 'original_r_image_%03d.jpg'%(0)))

        posterior_state = model.posterior_initial_update(video[:, :, 0])
        prior_state = None
        predictor_state = None

        skip = None
        prev_frames = video[:, :, 0]
        curr_frames = video[:, :, 1]

        for t in range(1, video.shape[2]):
            print(t)
            recon_frames, _, _, _, _, posterior_state, prior_state, predictor_state, skip \
                = model(prev_frames, curr_frames, posterior_state, prior_state, predictor_state, skip)
            recon_frames = make_grid(recon_frames, nrow=8).cpu().detach()
            original_frames = make_grid(curr_frames, nrow=8).cpu().detach()
            save_image(recon_frames, os.path.join(result_dir, 'reconstruction_image_%03d.jpg'%(t)))
            save_image(original_frames, os.path.join(result_dir, 'original_r_image_%03d.jpg'%(t)))
            if t < args.n_past:
                skip = None
            if t < video.shape[2] -1 :
                prev_frames = video[:, :, t]
                curr_frames = video[:, :, t+1]

        grid2gif(os.path.join(result_dir, 'reconstruction_image_*'), os.path.join(result_dir, 'reconstruction_image.gif'), delay=10)
        grid2gif(os.path.join(result_dir, 'original_r_image_*'), os.path.join(result_dir, 'original_r_image.gif'), delay=10)

        if i==0:
            break

if __name__ == '__main__':
    main()

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as T

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from tensorboardX import SummaryWriter

from model import SVP
from Options import TrainOptions
from dataset import StochasticMovingMNIST
from loss import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = TrainOptions()
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # transforms = T.Compose([T.ToTensor(),
    #                         MovingMNIST.normalize])
    transforms = T.ToTensor()

    # dataset = MovingMNIST(args.data_dir, transforms, True)
    dataset = StochasticMovingMNIST(True)
    dataloader = DataLoader(dataset, args.batch_size, True, pin_memory=True)

    logger = SummaryWriter(args.log_dir)

    model = SVP(args.img_dim, args.h_dim, args.g_dim, args.latent_dim, args.rnn_size,
                args.pre_n_layers, args.pos_n_layers, args.pri_n_layers).to(device)
    if os.path.exists(os.path.join(args.ckpt_dir, args.resume)):
        s_epoch, step = model.load(os.path.join(args.ckpt_dir, args.resume))
    else:
        s_epoch, step = -1, 0

    optim = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.l2_decay)

    for epoch in range(s_epoch, args.total_epoch):
        for i, video in enumerate(dataloader):
            step += 1
            video = video.to(device)
            recon_frames = [video[:4, :, 0]]

            posterior_state = model.posterior_initial_update(video[:, :, 0])
            prior_state = None
            predictor_state = None
            skip = None
            kld_loss = 0
            mse_loss = 0

            for t in range(1, args.n_past + args.n_future):
                prev_frames = video[:, :, t-1]
                curr_frames = video[:, :, t]
                predict_frames, mu, logvar, mu_p, logvar_p, posterior_state, prior_state, predictor_state, skip \
                    = model(prev_frames, curr_frames, posterior_state, prior_state, predictor_state, skip)
                recon_frames.append(predict_frames[:4])
                if t < args.n_past:
                    skip = None
                kld_loss += kl_div_loss(mu, logvar, mu_p, logvar_p)
                mse_loss += img_recon_loss(predict_frames, curr_frames)

            total_loss = kld_loss + mse_loss
            optim.zero_grad()
            total_loss.backward()
            optim.step()

            print("Epoch: %d/%d, Step: %d/%d, loss:%.4f"%(epoch+1, args.total_epoch, i+1, len(dataloader), total_loss.data.item()))

            if (step+1) % args.log_freq == 0:
                loss_dict = {'total loss': total_loss.data.item(),
                             'kld loss': kld_loss.data.item(),
                             'recon loss': mse_loss.data.item()}
                logger.add_scalars('loss', loss_dict, step)

                orig_vid = video[:4, :, :args.n_past+args.n_future].transpose(2, 3).reshape(4, args.img_dim, video.shape[3], -1)
                # orig_vid = make_grid(MovingMNIST.denormalize(orig_vid.cpu().detach()), nrow=1)
                # orig_vid = MovingMNIST.denormalize(make_grid(orig_vid, nrow=1).cpu().detach())
                orig_vid = make_grid(orig_vid, nrow=1).cpu().detach()

                recon_vid = torch.cat(recon_frames, dim=-1)
                # recon_vid = make_grid(MovingMNIST.denormalize(recon_vid.cpu().detach()), nrow=1)
                # recon_vid = MovingMNIST.denormalize(make_grid(recon_vid, nrow=1).cpu().detach())
                recon_vid = make_grid(recon_vid, nrow=1).cpu().detach()

                logger.add_image('original image', orig_vid, step)
                logger.add_image('reconstruced image', recon_vid, step)

        if (epoch + 1) % args.save_freq == 0:
            model.save(os.path.join(args.ckpt_dir, args.resume), epoch, step)
            model.save(os.path.join(args.ckpt_dir, 'model_%03d.ckpt'%(epoch+1)), epoch, step)

if __name__ == '__main__':
    main()












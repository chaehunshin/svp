import argparse

def TrainOptions():
    parser = argparse.ArgumentParser(description='train the svp with learned prior')

    parser.add_argument('--img_dim', default=1)
    parser.add_argument('--h_dim', default=128)
    parser.add_argument('--g_dim', default=128)
    parser.add_argument('--latent_dim', default=10)
    parser.add_argument('--rnn_size', default=256)
    parser.add_argument('--pre_n_layers', default=2)
    parser.add_argument('--pos_n_layers', default=1)
    parser.add_argument('--pri_n_layers', default=1)

    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--l2_decay', default=1e-5)
    parser.add_argument('--total_epoch', default=3000)
    parser.add_argument('--n_past', default=5)
    parser.add_argument('--n_future', default=10)

    parser.add_argument('--log_dir', default='/home/chaehuny/data/svp/smnist_logs')
    parser.add_argument('--ckpt_dir', default='/home/chaehuny/data/svp/smnist_ckpt')
    parser.add_argument('--data_dir', default='/home/chaehuny/data/dataset/moving_mnist/mnist_test_seq.npy')
    parser.add_argument('--resume', default='model_last.ckpt')
    parser.add_argument('--log_freq', default=20)
    parser.add_argument('--save_freq', default=10)
    args = parser.parse_args()
    return args

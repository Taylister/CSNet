gpu = 0

lamb = 10
#parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
epoch_count = 1
# parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
niter = 25
# parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
niter_decay = 25
# parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
lr = 0.0002
# parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
lr_policy = "lambda"
# parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
lr_decay_iters = 10
# parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
beta1 = 0.5
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
snap_period = 2500
save_ckpt_interval = 5000
write_log_interval = 100

# data
batch_size = 8
data_shape = [64, None]
data_dir = 'datasets/csnet_data'

train_data_dir = 'train'
test_data_dir = 'test'

i_s_dir = 'i_s'
mask_t_dir = 'mask_t'

example_data_dir = 'datasets/sample/train'
train_result_dir = 'result'

# train
train_ckpt_G_path = None
train_ckpt_D_path = None

# predict
predict_ckpt_path = "/home/miyazonotaiga/デスクトップ/MyResearch/CSNet/result/20201208160755/weight/netG_model_4-0.pth"
predict_data_dir = 'datasets/sample/train'
predict_result_dir = 'datasets/sample/train'
gpu = False
# Training settings
# parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# parser.add_argument('--dataset', required=True, help='facades')
# parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
# parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
# parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
# parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
# parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
# parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
# parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
# parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
# parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
# parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
# parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
# parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
# parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--cuda', action='store_true', help='use cuda?')
# parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
# parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
# opt = parser.parse_args()
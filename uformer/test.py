import numpy as np
import os, sys
import argparse
from tqdm import tqdm
import paddle.nn as nn
import paddle
from x2paddle.torch2paddle import DataLoader
import paddle.nn.functional as F
sys.path.append('/home/aistudio')
import scipy.io as sio
from utils.loader import get_validation_data, get_testA_data
import utils
from model import UNet
from model import Uformer
from model import Uformer_Cross
from model import Uformer_CatCross
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.get_device('cpu')

# from skimage import img_as_float32
# from skimage import img_as_ubyte
# from skimage.metrics import peak_signal_noise_ratio as psnr_loss
# from skimage.metrics import structural_similarity as ssim_loss
parser = argparse.ArgumentParser(description=\
    'RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default=\
    '/home/aistudio/demoire', type=str, help=\
    'Directory of validation images')
parser.add_argument('--result_dir', default='uformer/result_B',
    type=str, help='Directory for results')
parser.add_argument('--weights', default=
    '/home/aistudio/uformer/log/Uformer_/model_B/model_best.pdiparams', type=str, help=\
    'Path to weights')
parser.add_argument('--gpus', default='0', type=str, help=\
    'CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help=\
    'Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help=\
    'Save denoised images in result directory', default=True)
parser.add_argument('--embed_dim', type=int, default=32, help=\
    'number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help=\
    'number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help=\
    'linear/conv token projection')
parser.add_argument('--token_mlp', type=str, default='leff', help=\
    'ffn/leff token mlp')
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help=\
    'vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False,
    help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False,
    help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help
    ='share vit module')
parser.add_argument('--train_ps', type=int, default=256, help=\
    'patch size of training sample')
args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
utils.mkdir(args.result_dir)

testA_dataset = get_testA_data(args.input_dir)
testA_loader = DataLoader(dataset=testA_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

model_restoration= utils.get_arch(args)
# model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()
with paddle.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(testA_loader), 0):
        rgb_noisy = data_test[0]
        filenames = data_test[1]
        # print(filenames)
        h, w = rgb_noisy.shape[2], rgb_noisy.shape[3]

        rgb_restored = model_restoration(rgb_noisy)
        # print(rgb_restored)
        rgb_restored = rgb_restored * 255
        rgb_restored = paddle.clip(rgb_restored,0,255).cpu().numpy().squeeze().transpose((1,2,0))

        if args.save_images:
            utils.save_img(os.path.join(args.result_dir,filenames[0]), rgb_restored)

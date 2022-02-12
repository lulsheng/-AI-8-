import os
import sys
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))
print(dir_name)
import argparse
import options
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)
import utils
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader
import random
import time
import numpy as np
import datetime
from losses import CharbonnierLoss, SSIM, L1Loss
from warmup_scheduler import GradualWarmupScheduler
from utils.loader import get_training_data
from utils.loader import get_validation_data

use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.get_device('cpu')

log_dir = os.path.join(dir_name, 'log', opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat() + '.txt')
print('Now time is : ', datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models_ssim_l1')
utils.mkdir(result_dir)
utils.mkdir(model_dir)
random.seed(1234)
np.random.seed(1234)
paddle.seed(1234)
model_restoration = utils.get_arch(opt)

with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model_restoration) + '\n')

start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.
        lr_initial, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.
        weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer =optim.AdamW(parameters=model_restoration.parameters(),learning_rate=opt.lr_initial,
                           beta1=0.9, beta2=0.999, epsilon=1e-08, weight_decay=opt.weight_decay)
else:
    raise Exception('Error optimizer...')
# model_restoration = paddle.DataParallel(model_restoration)

if opt.resume:
    step = 10
    path_chk_rest = opt.pretrain_weights
    checkpoints = paddle.load(path_chk_rest)

    start_epoch = checkpoints['epoch']
    optimizer_state = checkpoints['optimizer']
    model_state = checkpoints['state_dict']
    optimizer.set_state_dict(optimizer_state)
    model_restoration.load_state_dict(checkpoints['state_dict'])

    new_lr = 0.0001
    print('------------------------------------------------------------------------------')
    print('==> Resuming Training with learning rate:', new_lr)
    print('------------------------------------------------------------------------------')
    scheduler = optim.lr.StepDecay(learning_rate=new_lr, step_size=step, gamma=0.5, verbose=False)
    print(scheduler.get_lr())
    # optimizer._learning_rate = scheduler
    optimizer.step()
if opt.warmup:
    print('Using warmup and cosine strategy!')
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr.CosineAnnealingDecay(eta_min=1e-06, T_max=opt.nepoch - warmup_epochs, learning_rate=0.01)
    optimizer._learning_rate = scheduler_cosine
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch =warmup_epochs, after_scheduler=scheduler_cosine)
    optimizer.step()
else:
    step = 10
    print('Using StepLR,step={}!'.format(step))
    scheduler = optim.lr.StepDecay(learning_rate=0.0001, step_size=step, gamma=0.5, verbose=False)
    optimizer._learning_rate = scheduler
    optimizer.step()

criterion1 = CharbonnierLoss()
criterion2 = SSIM()
criterion3 = L1Loss()
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.train_workers, drop_last=True)
val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.eval_workers, drop_last=True)
len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print('Sizeof training set: ', len_trainset, ', sizeof validation set: ', len_valset)
with paddle.no_grad():
    psnr_val_rgb = []
    for ii, data_val in enumerate(val_loader, 0):
        target = data_val[0]
        input_ = data_val[1]
        filenames = data_val[2]
        psnr_val_rgb.append(utils.batch_PSNR(input_, target, False).item())
    psnr_val_rgb = sum(psnr_val_rgb) / len_valset
    print('Input & GT (PSNR) -->%.4f dB' % psnr_val_rgb)
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader) // 8
print("""Evaluation after every {} Iterations !!!""".format(eval_now))
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    for i, data in enumerate(train_loader, 0):
        optimizer.clear_grad()
        target = data[0]
        input_ = data[1]
        # if epoch > 5:
        #     target, input_ = utils.MixUp_AUG().aug(target, input_)
        with paddle.amp.auto_cast():

            restored = model_restoration(input_)
            restored = paddle.clip(restored, 0, 1)
            # loss = criterion1(restored, target) # charbonnier
            loss2 = 1 - criterion2(restored, target) # ssim
            loss3 = criterion3(restored, target) # L1loss
            # print(loss1.item(), loss2.item())
            loss = loss2 + loss3
            # loss = loss1
            # print(loss.item())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if (i + 1) % eval_now == 0 and i > 0:
            with paddle.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                # ssim_val_rgb = []
                for ii, data_val in enumerate(val_loader, 0):

                    target = data_val[0]
                    input_ = data_val[1]
                    filenames = data_val[2]
                    with paddle.amp.auto_cast():
                        restored = model_restoration(input_)

                    restored = paddle.clip(restored, 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())
                    # ssim_val_rgb.append(utils.batch_SSIM(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb) / len_valset
                # ssim_val_rgb = sum(ssim_val_rgb) / len_valset

                if psnr_val_rgb > best_psnr and psnr_val_rgb <1000:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    paddle.save({'epoch': epoch, 'state_dict':model_restoration.state_dict(),
                                 'optimizer':optimizer.state_dict()}, os.path.join(model_dir,'model_best.pdparams'))
                    paddle.save(optimizer.state_dict(), os.path.join(model_dir,"optimizer_best.pdopt"))
                print('[Ep %d it %d\t PSNR demoire: %.4f\t] ----  [best_Ep_demoire %d best_it_demoire %d Best_PSNR_demoire %.4f LOSS %.4f] '% (epoch, i, psnr_val_rgb,best_epoch, best_iter,best_psnr, epoch_loss))
                with open(logname, 'a') as f:
                    f.write('[Ep %d it %d\t PSNR demoire: %.4f\t] ----  [best_Ep_demoire %d best_it_demoire %d Best_PSNR_demoire %.4f] ' \
                    % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')
                model_restoration.train()
    scheduler.step()

    print('------------------------------------------------------------------')
    print('Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}'.format(epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_lr()))
    print('------------------------------------------------------------------')
    with open(logname, 'a') as f:
        f.write('Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}'.format(epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_lr()) + '\n')
    paddle.save({'epoch': epoch, 'state_dict': model_restoration.state_dict()}, os.path.join(model_dir, 'model_latest.pdparams'))
    paddle.save(optimizer.state_dict(), os.path.join(model_dir,"optimizer_best.pdopt"))
    if epoch % opt.checkpoint == 0:
        paddle.save({'epoch': epoch, 'state_dict': model_restoration.state_dict()},
                     os.path.join(model_dir, 'model_epoch_{}.pdparams'.format(epoch)))
        paddle.save(optimizer.state_dict(), os.path.join(model_dir, "optimzer_epoch_{}.pdopt".format(epoch)))
print('Now time is : ', datetime.datetime.now().isoformat())

import paddle
import numpy as np
import pickle
import cv2


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in ['.npy'])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg'])


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in ['.png'])


def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in ['.pkl'])


def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict


def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)


def load_npy(filepath):
    img = np.load(filepath)
    return img


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    return img


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def myPSNR(tar_img, prd_img):
    imdff = paddle.clip(prd_img, 0, 1) - paddle.clip(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20*paddle.log10(1/rmse)
    return ps


def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR) / len(PSNR) if average else sum(PSNR)

def mySSIM(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = np.array(img1)
    img2 = np.array(img2)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def batch_SSIM(img1, img2, average=True):
    SSIM = []
    for im1, im2 in zip(img1, img2):
        psnr = mySSIM(im1, im2)
        SSIM.append(psnr)
    return sum(SSIM) / len(SSIM) if average else sum(SSIM)
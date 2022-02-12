import os
import cv2
import numpy as np
import math

test_fileDir = '/home/aistudio/moire_testB_dataset'  # 原始分辨率的数据集
test_file_list = sorted(os.listdir(test_fileDir))
result_fileDir = '/home/aistudio/uformer/uformer/result_B'  # 从模型的得到的结果
output_fileDir = './result_B_final' # 最终得到的原始分辨率的结果
# result_file_list = sorted(os.listdir(result_fileDir), key=lambda x:(x.split('_')[0],x.split('_')[1].split('.')[0]))
result_file_list = os.listdir(result_fileDir)
# print(result_file_list)
result_file_list = [i for i in os.listdir(result_fileDir) if os.path.splitext(i)[-1] == ".jpg"]
result_file_list.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))


if not os.path.exists(output_fileDir):
    os.mkdir(output_fileDir)

flag = 0
for i, file in enumerate(test_file_list, 1):
    file_path = os.path.join(test_fileDir, file)
    image = cv2.imread(file_path) # h,w,c
    h , w = image.shape[:2]
    num_patch = math.ceil(max(h, w) / 1664)
    l = 1664 * num_patch
    image_middle = np.zeros((l, l, 3), np.uint8)
    for j in range(0, num_patch):
        for m in range(0, num_patch):
            img = os.path.join(result_fileDir, '{}_{}.jpg'.format(i,j*num_patch+m+1))
            print(j,m,img)
            image_middle[m*1664: (m+1)*1664,1664*j: 1664*(j+1),  :] = cv2.imread(img)
    img_final = image_middle[:h,:w,:]
    cv2.imwrite(os.path.join(output_fileDir, 'moire_testB_{:0>5d}.jpg'.format(i-1)),img_final)

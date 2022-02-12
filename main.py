#!/usr/bin/env python
# coding: utf-8

# # 百度网盘AI大赛——图像处理挑战赛：文档图像摩尔纹消除8名方案

# ## 一、赛题介绍与分析
# 1. **任务**: 对含有摩尔纹的低质图片进行图像复原，去除其中的摩尔纹
# 
# 2. **分析**: 摩尔纹源于法文Moiré一词,泛指两个频率相近的等幅正弦波叠加而产生的干涉条纹。在现代数字成像过程中,由于图像采集设备的采样率有限,某些场景下采集图像极易出现摩尔纹效应,如针织布料的拍摄图像、LED屏幕的屏摄图像等,严重影响成像质量和后续的图像分析处理。图像摩尔纹属于欠采样导致的信号混叠干扰,与拍摄场景密切相关,因此摩尔纹降质图像的恢复算法研究工作具有很大的挑战性。<br>
# 由于图片的摩尔纹并不总是存在于全局区域，而是在更多情况下存在于局部区域中。所以如何利用局部上下文信息进行图像复原显得尤为重要。由于标准Transformer在所有词之间计算全局自注意力，而在捕获局部上下文信息方面比较弱，所以我们使用了局部增强窗口Transformer ，既利用了Transformer的自注意力机制以捕获长距离依赖，又同时将卷积引入Transformer以捕获有用的局部上下文信息。
# 
# ## 二、数据集处理
# 1. **额外数据集的使用**： 比赛所给的训练集只有1000张图片，我们增加了AIM 2019数据集中的validation dataset的100张图片增加了数据集的丰富度。
# 2. **数据增强（训练集）**：由于训练集中的图像的分辨率都是各种各样的，我们把所有图片都随机切割成256x256分辨率的正方形，每一张图片及那才成32张，这样我们就有三万多张的图片，然后输入网络中进行训练。
# 3. **数据增强（测试集）**：测试的时候，我们先把所有测试图片padding成1664x1664的分分辨率。然后输入网络中进行测试，之后按照原始尺寸进行剪裁，得到最终的结果图像。
# 
# ## 三、模型介绍
# 1. 因为一般的基于CNN的网络结构对全局信息的捕捉不够充分，而摩尔纹的结构信息十分
# 依赖于全局信息。我们采用transformer来实现我们的网络。基于之前U-Net在low-level图像处理中的应用。我们采用一种U-shape的transformer结构来捕捉全局依赖关系。总体结构如下图：
# ![image.png](attachment:fc32fb96-a9e2-4c51-a8f0-447d2e21be12.png) 
# 设计了如下模块：
# **Locally-enhanced Window Transformer block (LeWin)**: Non-overlapping window-based self-attention instead of global self- attention.减少了全局self-attention的计算量，捕捉了局部注意力。<br>
# **Window-based Multi-head Self-Attention (W-MSA)**
#     **Locally-enhanced Feed-Forward Network (LeFF)**.
# **Multi-Scale Restoration Modulator**
# 细节参见论文《Uformer: A General U-Shaped Transformer for Image Restoration》
# 
# ## 四、实现细节
# 1. Loss: L1Loss + SSIM Loss
# 2. 一共训练60 epoch， 其中第55个epoch的效果最好
# 
# ## 五、代码运行
# 在 /uformer/options.py文件中设置超参数。
# 首先运行train.py文件进行训练，再运行test.py文件进行测试
# 最后运行crop.py文件对得到的结果进行剪裁，以得到最终结果。
# 
# ## 参考
# 1. 论文 《Uformer: A General U-Shaped Transformer for Image Restoration》
# 2. 代码 https://github.com/ZhendongWang6/Uformer

# In[3]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[1]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[9]:


get_ipython().run_line_magic('cd', '/home/aistudio/uformer/')
get_ipython().system('python train.py')


# In[1]:


get_ipython().system('ls data/data126672')
get_ipython().system('unzip data/data126672/moire_testB_dataset.zip')


# In[7]:


get_ipython().run_line_magic('cd', '/home/aistudio/uformer/')
get_ipython().system('python test.py')
# 得到结果uformer/uformer/result_B/分辨率为1664x1664


# In[3]:


get_ipython().run_line_magic('cd', '/home/aistudio/uformer')
get_ipython().system('python crop.py')


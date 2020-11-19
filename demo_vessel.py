import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec#调用网格
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
opt = parser.parse_args()

data_dir = opt.data_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat('./model/%s/pytorch_result.mat'%opt.name)
query_feature = torch.FloatTensor(result['query_f'])
# query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
# gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]


query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, gf, gl):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    return index

i = opt.query_index
index = sort_img(query_feature[i],query_label[i],gallery_feature,gallery_label)

########################################################################
# Visualize the rank result

query_path, _ = image_datasets['query'].imgs[i]
query_label = query_label[i]
print(query_path)
print('Top 10 images are as follow:')
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    

    fig = plt.figure(figsize=(20,4))
    gs=gridspec.GridSpec(2,7)#设定网格
    # ax = plt.subplot(1,6,1)
    ax = fig.add_subplot(gs[:,0:2])#选定网格
    # fig = plt.figure(figsize=(16,4))
    # ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    for i in range(5):
        ax = plt.subplot(2,7,i+3)#从第3个图开始画，但是i从0开始
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_label:
            # ax.set_title('%d'%(i+1), color='green')
            for child in ax.get_children():
                if isinstance(child, matplotlib.spines.Spine):
                    child.set_color('green')
                    child.set_linewidth(3)

        else:
            # ax.set_title('%d'%(i+1), color='red')
            for child in ax.get_children():
                if isinstance(child, matplotlib.spines.Spine):
                    child.set_color('red')
                    child.set_linewidth(3)
        print(img_path)
    for i in range(5,10):
        ax = plt.subplot(2,7,i+5)
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_label:
            # ax.set_title('%d'%(i+1), color='green')
            for child in ax.get_children():
                if isinstance(child, matplotlib.spines.Spine):
                    child.set_color('green')
                    child.set_linewidth(3)

        else:
            # ax.set_title('%d'%(i+1), color='red')
            for child in ax.get_children():
                if isinstance(child, matplotlib.spines.Spine):
                    child.set_color('red')
                    child.set_linewidth(3)
        print(img_path)
    fig.tight_layout()#调整整体空白
    plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
dir_name = "./model/%s/demo"%opt.name
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
fig.savefig(dir_name + "/%s.png"%(opt.query_index),bbox_inches='tight',pad_inches=0.1)

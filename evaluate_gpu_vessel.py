"""  
输入：pytorch_result.mat
操作：计算CMC曲线，rank和map
输出：输出rank和map，写入result.txt
"""
import scipy.io
import torch
import numpy as np
#import time
import os
import argparse

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
opt = parser.parse_args()


str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
#######################################################################
# Evaluate
def evaluate(qf,ql,gf,gl):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    ap_CMC = compute_mAP(index, query_index)
    return ap_CMC


def compute_mAP(index, query_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if query_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # find good_index index
    ngood = len(query_index)
    mask = np.in1d(index, query_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
result = scipy.io.loadmat('./model/%s/pytorch_result.mat'%opt.name)

query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]


query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],gallery_feature,gallery_label)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    #print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('mAP:%f Rank-1:%f Rank-5:%f Rank-10:%f '%(ap/len(query_label),CMC[0],CMC[4],CMC[9]))

result = './model/%s/result.txt'%opt.name

with open(result, 'w') as f:
    f.write('mAP:%f Rank-1:%f Rank-5:%f Rank-10:%f '%(ap/len(query_label),CMC[0],CMC[4],CMC[9]))
    f.write(CMC)



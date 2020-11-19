"""  
evaluate_gpu_vessel.py的排序版本
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
import pandas as pd

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
    index = index[::-1]#size为126474，即gallery的size，经过score排序的 
    #index是根据qf和gf计算出来的score从大到小排序的 gl的下标，即预测值。
    # index = index[0:2000]#在这里切片可以只计算前一部分的rank和map
    # good index
    query_index = np.argwhere(gl==ql)#在gl里找和ql相同的index，即query_index是GT
    ap_CMC = compute_mAP(index, query_index)#预测值和gt
    return ap_CMC


def compute_mAP(index, query_index):
    # In fact, the value of mAP only changed when it meets the true-matches
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if query_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # find good_index index
    ngood = len(query_index)
    mask = np.in1d(index, query_index)#比对哪里相同
    rows_good = np.argwhere(mask==True)#得到相同的下标
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1#从找到的第一个开始，后面全部置1，也就是说单张图片的查询，rank只有0或1
    # the value of mAP only changed when it meets the true-matches。
    # ngood是当前查询图片的gt的数量
    for i in range(ngood):
        d_recall = 1.0/ngood#recall是每一轮固定的，分成1/gt份
        precision = (i+1)*1.0/(rows_good[i]+1)#第i个正确返回图片/第i个返回图片
        if rows_good[i]!=0:#不是0
            old_precision = i*1.0/rows_good[i]#第i-1个正确返回图片/第i-1个返回图片
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2#precision和old_precision的平均值，梯形的面积

    return ap, cmc

######################################################################
result = scipy.io.loadmat('./model/%s/pytorch_result.mat'%opt.name)

query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
query_filename = result['query_filename'][0]
# query_label = result['query_label'][0][:50]#切片，只看前50张
# query_filename = result['query_filename'][0][:50]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]


query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
ap_list = []
CMC_list = []
#print(query_label)
# 查询的循环 
for i in range(len(query_label)):#每一个query_label会得出一个ap_tmp和CMC_tmp
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],gallery_feature,gallery_label)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp# 维度是1x126474
    ap += ap_tmp # 这里ap是总和，最后输出才除以len(query_label)
    ap_list.append(ap_tmp)
    CMC_list.append(CMC_tmp)
    #print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
ap = ap/len(query_label) #average ap
print('mAP:%f Rank-1:%f Rank-5:%f Rank-10:%f '%(ap,CMC[0],CMC[4],CMC[9]))

result = './model/%s/result.txt'%opt.name

with open(result, 'w') as f:
    f.write('mAP:%f Rank-1:%f Rank-5:%f Rank-10:%f '%(ap,CMC[0],CMC[4],CMC[9]))

# CMC = np.array(CMC)
# CMC_df = pd.DataFrame(CMC_list)
# CMC_df.columns = ['CMC']
# CMC_df = CMC_df.sort_values(by = 'CMC')

# CMC = np.around(CMC, 5)
# CMC_path = './model/%s/CMC.csv'%opt.name
# np.savetxt(CMC_path,CMC,fmt='%.04f')
# CMC_df.to_csv(CMC_path)


ap_df = pd.DataFrame(ap_list)
ap_path = './model/%s/ap.csv'%opt.name
ap_df.index.name='query_index'
ap_df.columns = ['ap']
ap_df.insert(0, 'filename',query_filename)
ap_df.insert(1, 'label',query_label)
ap_df['label'] = ap_df['label'].astype(str)
ap_df['filename'] = ap_df['filename'].astype(str)

query_path = "/home/zlm/dataset/vessel_reid/pytorch/query/"
filepath = query_path + ap_df['label'] + '/' + ap_df['filename'] + '.jpg'

ap_df.insert(3, 'filepath',filepath)
ap_df = ap_df.sort_values(by = 'ap')
ap_df.to_csv(ap_path,columns=['label','filename','ap'])
filepath_path = './model/%s/filepath.csv'%opt.name
ap_df.to_csv(filepath_path,columns=['filepath'],index=0)
# ap_df.to_csv(ap_path,columns=['filepath'],index=0)


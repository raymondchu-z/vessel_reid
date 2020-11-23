""" 
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* We do not use it 
│   ├── gt_query/                   /* Files for multiple query testing 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
│   ├── pytorch/
│       ├── train/                   /* train 
│           ├── 0002
|           ├── 0007
|           ...
│       ├── val/                     /* val
│       ├── train_all/               /* train+val      
│       ├── query/                   /* query files  
│       ├── gallery/                 /* gallery files 
"""
import os
import shutil
import random
import re #用正则表达来分离括号
import pandas as pd
import numpy as np

def file_filter(f):
    if f[-4:] in ['.jpg', '.png', '.bmp']:
        return True
    else:
        return False
# You only need to change this line to your dataset download path
all_IMO_path = '/home/zlm/dataset/vessel_reid/ALL-IMG'
dateset_path = '/home/zlm/dataset/vessel_reid'

if not os.path.isdir(all_IMO_path):
    print('please change the download_path')

save_path = dateset_path + '/pytorch2'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

#---------------------------------------
#get trainval_list and test_list
#从trainval.txt和test.txt中读取数据，如Cargo,[8504478]，将IMO号储存在trainval_list和test_list

trainval_list = []
trainval_file = open(dateset_path + "/trainval.txt","r")
trainval_lines = trainval_file.readlines()
for eachLine in trainval_lines:
    trainval_lines_list = re.split(r",(?![^[\]]*\])", eachLine)
    IMOs = trainval_lines_list[1]
    IMOs = IMOs.rstrip().replace('[','').replace(']','').replace('\'','').replace(' ','')
    IMO_list = IMOs.split(',')
    trainval_list.extend(IMO_list)
trainval_file.close()

test_list = []
test_file = open(dateset_path + "/test.txt","r")
test_lines = test_file.readlines()
for eachLine in test_lines:
    test_file_list = re.split(r",(?![^[\]]*\])", eachLine)
    IMOs = test_file_list[1]
    IMOs = IMOs.rstrip().replace('[','').replace(']','').replace('\'','').replace(' ','')
    IMO_list = IMOs.split(',')
    test_list.extend(IMO_list)
test_file.close()
print("get trainval_list and test_list\n")
#---------------------------------------
#train_all
#根据trainval_list将一个IMO号下的文件复制到目标文件夹

trainval_save_path = save_path + '/train_all'

if not os.path.isdir(trainval_save_path):
    os.mkdir(trainval_save_path)

for IMO_dir in trainval_list:
    src_path = os.path.join(all_IMO_path,IMO_dir)
    dst_path = os.path.join(trainval_save_path,IMO_dir)
    if os.path.isdir(dst_path):
        shutil.rmtree(dst_path) #指的是IMO文件夹
    shutil.copytree(src_path, dst_path)
print("train_all cone\n")
#---------------------------------------
#train_val
#根据train_all里的文件，分成train和val，val是每个IMO下的一张
train_save_path = save_path + '/train'
val_save_path = save_path + '/val'

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)


if os.path.exists(trainval_save_path):
    dirs = os.listdir( trainval_save_path )#IMO文件夹
    for dir in dirs:
        files = os.listdir( os.path.join(trainval_save_path,dir ))#每张图片，注意会有csv文件！
        files = list(filter(file_filter, files))#筛选jpg
        src_path = trainval_save_path + '/' + dir
        for file in files:
            dst_path = train_save_path + '/' + dir #放在循环里面每次刷新
            if not os.path.isdir(dst_path):#第一次进入
                os.mkdir(dst_path)
                dst_path = val_save_path + '/' + dir#first image is used as val image 因为在一个循环里面只有第一张图片会出现没有dst_path的情况。在这里修改一次dst_path只会影响第一张图片。所以这张图片是不包括在train里面的。
                os.mkdir(dst_path)
            src_file = os.path.join(src_path,file)
            dst_file = os.path.join(dst_path,file)
            shutil.copyfile(src_file, dst_file)
print("train_val done\n")
#-----------------------------------------
#gallery and query
#根据test_list找到IMO,再根据file_seleted_label.csv分成两部分，因为有的文件夹下没有file_label.csv
#

gallery_save_path = save_path + '/gallery'
query_save_path = save_path + '/query'

if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)
    os.mkdir(query_save_path)

for IMO_dir in test_list:
    IMO_path = os.path.join(all_IMO_path,IMO_dir)
    query_df = pd.read_csv(IMO_path +"/file_seleted_label.csv")
    # all_df = pd.read_csv(src_path +"/file_label.csv")
    # gallery_df = all_df.append(query_df).drop_duplicates(keep=False)
    # gallery_list = gallery_df['filename'].value_counts().index.tolist()
    query_list = query_df['filename'].value_counts().index.tolist()
    files_list = os.listdir( IMO_path )
    files_list = list(filter(file_filter, files_list))
    gallery_list = list(set(files_list)^set(query_list))
    for name in gallery_list:
        src_path = os.path.join(IMO_path, name)
        dst_path = os.path.join(gallery_save_path,IMO_dir)
        if not os.path.isdir(dst_path):#第一次进入
            os.mkdir(dst_path)
        shutil.copyfile(src_path, dst_path + "/" +name)
    for name in query_list:
        src_path = os.path.join(IMO_path, name)
        dst_path = os.path.join(query_save_path,IMO_dir)
        if not os.path.isdir(dst_path):#第一次进入
            os.mkdir(dst_path)
        shutil.copyfile(src_path, dst_path + "/" +name)
print("gallery and query done\n")
        



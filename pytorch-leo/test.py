############################################################################################
# !/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Author  : zhaoqinghui
# Date    : 2017.1.11
# Function:  1.生成train.Thumbs.db
#           2.生成label.Thumbs.db
#           3.将train.Thumbs.db 和label.Thumbs.db合并成train.pkl
##########################################################################################

import os
import _pickle as cPickle
from PIL import Image
import numpy as np

'''
###############################################################
'''
# flag='train'
flag = 'test'
filename = 'C:/Users/14557/Desktop/data1'
outfile = 'C:/Users/14557/Desktop/embeddings/tieredImageNet/center/train_embeddings.pkl'

# filename = 'C:/Users/14557/Desktop/data2'
# outfile = 'C:/Users/14557/Desktop/embeddings/tieredImageNet/center/test_embeddings.pkl'

# filename = 'C:/Users/14557/Desktop/data3'
# outfile = 'C:/Users/14557/Desktop/embeddings/tieredImageNet/center/val_embeddings.pkl'
imgsize = 32
'''
################################################################
'''


# 获得文件夹下所有图片
def getFilePicture(folder):
    assert os.path.exists(folder)
    assert os.path.isdir(folder)
    PictureList = os.listdir(folder)
    PictureList = [str(folder) + '/' + item for item in PictureList]
    return PictureList


# 转换函数
def writepkltemp(rootDir):
    # all_data = []
    # all_lable = []
    # all_keys = []
    for lists in os.listdir(rootDir):
        label = 1 if lists == 'MeanderControl' else 0

        for pic in os.listdir(os.path.join(rootDir, lists)):
            path = os.path.join(rootDir, lists) + '/' + pic

            # print(path, label)
            img = Image.open(path).convert("L")
            img = img.resize((imgsize, imgsize))
            # img = img.resize(-1)
            # if img.size > 2:
            #     img = img.convert("1")
            img_ndarray = np.asarray(img, dtype='float32') / 256
            # print img_ndarray.shape
            global vector
            global vector_label
            global vectot_path
            global num
            # print len(np.ndarray.flatten(img_ndarray))
            vector[num] = np.ndarray.flatten(img_ndarray)
            vector_label[num] = int(label)
            # vector_path.append(path)
            key = pic.replace('-', '')

            vector_path[num] = 'kkkkk-{}-{}'.format(lists, key)

            # print(vector_path)
            num = num + 1

            # all_data.append(img_ndarray)
            # all_lable.append(int(label))
            # all_keys.append(path)


# 保存pkl格式图片集
def writepkl(filename):
    writepkltemp(filename)


# 获得文件夹下所有图片的数量
def getnum(rootDir):
    for lists in os.listdir(rootDir):
        path = rootDir + '/' + lists
        if os.path.isdir(path):
            getnum(path)
        else:
            global n
            n = n + 1


# 图片总量
def getimgnum(numpath):
    getnum(numpath)


if __name__ == "__main__":
    num = 0
    n = 0
    numpath = filename
    getimgnum(numpath)
    vector = np.empty((n, imgsize * imgsize),dtype=np.float32)
    vector_label = np.empty(n,dtype=np.int64)
    # vector_path = []
    vector_path = np.empty(n,dtype=object)


    # dic = dict()
    # dic['embeddings'] = np.array(vector)
    # dic['label'] = np.array(vector_label)
    # dic['keys'] = np.array(vectot_path)

    writepkl(filename)

    # vector_label = vector_label.astype(np.int)
    write_file = open(outfile, 'wb')
    a={"embeddings":vector[0:n],'labels':vector_label[0:n],'keys':vector_path[0:n]}
    # cPickle.dump([vector[0:n], vector_label[0:n]], write_file, -1)
    cPickle.dump(a, write_file, -1)
    write_file.close()

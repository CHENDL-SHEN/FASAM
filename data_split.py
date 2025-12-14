
"""
对 pseudo labels 的SC指标评价（21TMM-SALNet）;

pseudo labels：像素值为 0-20的伪标注

saliency map：二值； 1）经过 dCRF处理、二值化的显著性图；
                   2）这里使用DRS论文中用的显著性图，将gt二值化后与其计算miou为77%左右；
                   而EPS中使用的显著性图的miou仅为68%；但我的硕士大论文中使用的是EPS

"""


import cv2
from PIL import Image
import numpy as np
import torch
import xml.etree.ElementTree as ET
import os
import shutil
from pathlib import Path


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def add_txt(path, string):
    with open(path, 'a+') as f:
        f.write(string + '\n')

def log_print(message, path):
    """This function shows message and saves message.
    
    Args:
        pred_tags: 
            The type of variable is list.
            The type of each element is string.
        
        gt_tags:
            The type of variable is list.
            the type of each element is string.
    """
    print(message)
    add_txt(path, message)



### function for our private bu image
def get_K_fold_cross_validation():
    import numpy as np
    from sklearn.model_selection import KFold

    # 读取数据
    def read_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # 将数据转换为numpy数组
        data = np.array([line.strip().split() for line in lines], dtype=object)
        return data

    # 写入数据到文件
    def write_data(data, file_path):
        with open(file_path, 'w') as file:
            for line in data:
                file.write(' '.join(line) + '\n')

    # 文件路径
    file_path = '/media/ders/XS/dataset/BUSD/Private_BUTS_with_GT/imageid.txt'  # 替换为你的txt文件路径

    # 读取数据
    data = read_data(file_path)

    # 初始化KFold对象，设置为5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 执行5折交叉验证的数据分割
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        # 获取训练集和测试集的数据
        train_data = list(data[train_index])
        test_data = list(data[test_index])
        
        # 为每一折创建文件名
        train_file = f'/media/ders/XS/dataset/BUSD/Private_BUTS_with_GT/fold_{fold+1}_train.txt'
        test_file = f'/media/ders/XS/dataset/BUSD/Private_BUTS_with_GT/fold_{fold+1}_test.txt'
        
        # 写入训练集和测试集数据到文件
        write_data(train_data, train_file)
        write_data(test_data, test_file)
        
        print(f"Fold {fold+1} - Training set size: {len(train_data)}, Test set size: {len(test_data)}")


def get_split_dataset():
    import numpy as np
    from sklearn.model_selection import KFold

    # 读取数据
    def read_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # 将数据转换为numpy数组
        data = [line.strip().split() for line in lines]
        return data

    # 写入数据到文件
    def write_data(data, file_path):
        with open(file_path, 'w') as file:
            for line in data:
                file.write(line + '\n')

    splitL = ['train', 'test']
    idL = ['1','2','3','4','5']

    for s in splitL:
        for j in idL:
            # 文件路径
            file_path = '/media/ders/XS/dataset/BUSD/Private_BUTS_with_GT/fold_%s_%s.txt' % (j, s)  # 替换为你的txt文件路径

            # 读取数据
            data = read_data(file_path)
            data_l = []
            data_v = []

            for i in range(len(data)):
                if 'benign' in data[i][0]:
                    data_a = 'benign_%s 0' % data[i][0].split('/')[3]
                    # data_a = 'benign_%s' % (data[i][0].split('/')[3])
                    data_l.append(data_a)

                if 'inflammation' in data[i][0]:
                    data_a = 'inflam_%s 1' % data[i][0].split('/')[3]
                    data_l.append(data_a)

                if 'malign' in data[i][0]:
                    data_a = 'malign_%s 2' % data[i][0].split('/')[3]
                    data_l.append(data_a)
            
            for i in range(len(data)):
                if 'benign' in data[i][0]:
                    data_a = 'benign_%s' % data[i][0].split('/')[3]
                    # data_a = 'benign_%s' % (data[i][0].split('/')[3])
                    data_v.append(data_a)

                if 'inflammation' in data[i][0]:
                    data_a = 'inflam_%s' % data[i][0].split('/')[3]
                    data_v.append(data_a)

                if 'malign' in data[i][0]:
                    data_a = 'malign_%s' % data[i][0].split('/')[3]
                    data_v.append(data_a)
                
            # 为每一折创建文件名
            train_file = '/media/ders/XS/dataset/BUSD/Private_BUTS_with_GT/ImageLists/IDCls/fold_%s_%s_ID.txt' % (j, s)
            train_filev = '/media/ders/XS/dataset/BUSD/Private_BUTS_with_GT/ImageLists/ID/fold_%s_%s_ID.txt' % (j, s)
            
            # 写入训练集和测试集数据到文件
            write_data(data_l, train_file)
            write_data(data_v, train_filev)


def get_clsnpy():
    
    data_dict = {}

    def one_hot_embedding(label, classes):
        """Embedding labels to one-hot form.

        Args:
        labels: (int) class labels.
        num_classes: (int) number of classes.

        Returns:
        (tensor) encoded labels, sized [N, #classes].
        """
        
        vector = np.zeros((classes), dtype = np.float32)
        vector[label] = 1.
        return vector 

    # 文件路径
    file_path1 = '/media/ders/XS/dataset/BUSD/Private_BUTS_with_GT/ImageLists/IDCls/fold_1_train_idcls.txt'  # 替换为你的txt文件路径
    file_path2 = '/media/ders/XS/dataset/BUSD/Private_BUTS_with_GT/ImageLists/IDCls/fold_1_test_idcls.txt'  # 替换为你的txt文件路径
    with open(file_path1, 'r') as file:
        data1 = [line.strip().split() for line in file.readlines()]
    
    with open(file_path2, 'r') as file:
        data2 = [line.strip().split() for line in file.readlines()]

    data = data1 + data2
    print(len(data))

    for d in data:
        ot = one_hot_embedding(int(d[1]), 3)  # 3是前景类别数，不包括背景
        data_dict[d[0]] = ot

    np.save(os.path.join('/media/ders/XS/dataset/BUSD/Private_BUTS_with_GT/ImageLists/ID/cls_labels.npy'), data_dict) 


### function for public BUSI dataset 
def convert_BUSI_to_ourfoldertype():
    import os
    import shutil

    beginpath = '/media/ders/XS/dataset/BUSD/Public_BUSI_with_GT/OriData/benign/'

    for img_name in os.listdir(beginpath):
        # print(img_name)

        if 'mask' in img_name:
            idx = img_name.split('_mask.png')[0].split(' (')[1].split(')')[0]
            print(idx)
            print(img_name)
            dstp = '/media/ders/XS/dataset/BUSD/Public_BUSI_with_GT/label_3/malign_%s.png' % idx
            shutil.copy(beginpath + img_name, dstp)
        
        if 'mask' not in img_name:
            idx = img_name.split('.png')[0].split(' (')[1].split(')')[0]
            print(idx)
            print(img_name)
            dstp = '/media/ders/XS/dataset/BUSD/Public_BUSI_with_GT/img/benign_%s.png' % idx
            shutil.copy(beginpath + img_name, dstp)


def get_semmask():
    beginpath = '/media/ders/XS/dataset/BUSD/Public_BUSI_with_GT/label_3_255/'
    save_path_inv = '/media/ders/XS/dataset/BUSD/Public_BUSI_with_GT/label_3/'
    
    flag = 0
    for img_name in os.listdir(beginpath):
        if 'benign' in img_name:
            
            mask = np.asarray(Image.open(beginpath + img_name), dtype=np.int64)
            shape = mask.shape
            mask = mask.reshape(-1)
            numBi = mask.size

            for i in range(numBi):
                if mask[i] == True:
                    mask[i] = 1
            mask = mask.reshape(shape)
        
            cv2.imwrite(os.path.join(save_path_inv, img_name), mask)  
        
        if 'malign' in img_name:
            
            mask = np.asarray(Image.open(beginpath + img_name), dtype=np.int64)
            shape = mask.shape
            mask = mask.reshape(-1)
            numBi = mask.size

            for i in range(numBi):
                if mask[i] == True:
                    mask[i] = 2
            mask = mask.reshape(shape)
        
            cv2.imwrite(os.path.join(save_path_inv, img_name), mask)  

            flag += 1
            print(flag)


def get_BUSI_splittxt():
    beginpath = '/media/ders/sdd1/XS/pipeline/dataset/BUS/PubBUSI/img/'
    save_path_IDCLs = '/media/ders/sdd1/XS/pipeline/dataset/BUS/PubBUSI/img/ALL_IDCls.txt'
    save_path_ID = '/media/ders/sdd1/XS/pipeline/dataset/BUS/PubBUSI/img/ALL_ID.txt'

    
    for img_name in os.listdir(beginpath):

        print(img_name)

        if 'benign' in img_name:
            log_print(img_name[:-4] + ' 0', save_path_IDCLs)

        if 'malign' in img_name:
            log_print(img_name[:-4] + ' 1', save_path_IDCLs)

        log_print(img_name[:-4], save_path_ID)


def get_DB_splittxt():
    beginpath = '/media/ders/sdd1/XS/pipeline/dataset/BUS/Public_DB_v2/img/'
    save_path_IDCLs = '/media/ders/sdd1/XS/SPCAM_FAMS/data/PubDB/ALL_IDCls.txt'
    save_path_ID = '/media/ders/sdd1/XS/SPCAM_FAMS/data/PubDB/ALL_ID.txt'

    
    for img_name in os.listdir(beginpath):

        print(img_name)

        if 'benign' in img_name:
            log_print(img_name[:-4] + ' 0', save_path_IDCLs)

        if 'malign' in img_name:
            log_print(img_name[:-4] + ' 1', save_path_IDCLs)

        log_print(img_name[:-4], save_path_ID)


def get_K_fold_cross_validation():
    import numpy as np
    from sklearn.model_selection import KFold

    # 读取数据
    def read_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # 将数据转换为numpy数组
        data = np.array([line.strip().split() for line in lines], dtype=object)
        return data

    # 写入数据到文件
    def write_data(data, file_path):
        with open(file_path, 'w') as file:
            for line in data:
                file.write(' '.join(line) + '\n')

    # 文件路径
    file_path = '/media/ders/sdd1/XS/SPCAM_FAMS/data/PubDB/ALL_ID.txt'  # 替换为你的txt文件路径

    # 读取数据
    data = read_data(file_path)

    # 初始化KFold对象，设置为5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 执行5折交叉验证的数据分割
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        # 获取训练集和测试集的数据
        train_data = list(data[train_index])
        test_data = list(data[test_index])
        
        # 为每一折创建文件名
        train_file = f'/media/ders/sdd1/XS/SPCAM_FAMS/data/PubDB/fold_{fold+1}_train.txt'
        test_file = f'/media/ders/sdd1/XS/SPCAM_FAMS/data/PubDB/fold_{fold+1}_test.txt'
        
        # 写入训练集和测试集数据到文件
        write_data(train_data, train_file)
        write_data(test_data, test_file)
        
        print(f"Fold {fold+1} - Training set size: {len(train_data)}, Test set size: {len(test_data)}")


def comput_class_balance():
    # txtpath = '/media/ders/XS/dataset/BUSD/Public_BUSI_with_GT/ImageLists/fold_1_train.txt'
    txtpath = '/media/ders/sdd1/XS/SPCAM_FAMS/data/PubDB/fold_2_test.txt'
    image_id_list = [image_id.strip() for image_id in open(txtpath).readlines()]

    benign = 0
    malign = 0
    inflam = 0

    for idx in image_id_list:
        if 'benign' in idx:
            benign += 1
        
        if 'malign' in idx:
            malign += 1
        
        if 'inflam' in idx:
            inflam += 1

    print(benign)
    print(malign)
    print(inflam)


def get_clsnpy_BUSI():
    
    data_dict = {}

    def one_hot_embedding(label, classes):
        """Embedding labels to one-hot form.

        Args:
        labels: (int) class labels.
        num_classes: (int) number of classes.

        Returns:
        (tensor) encoded labels, sized [N, #classes].
        """
        
        vector = np.zeros((classes), dtype = np.float32)
        vector[label] = 1.
        return vector 

    # 文件路径
    file_path = '/media/ders/sdd1/XS/SPCAM_FAMS/data/PubDB/ALL_IDCls.txt'  # 替换为你的txt文件路径
    with open(file_path, 'r') as file:
        data = [line.strip().split() for line in file.readlines()]

    print(len(data))

    for d in data:
        ot = one_hot_embedding(int(d[1]), 2)  # 2是前景类别数，不包括背景
        data_dict[d[0]] = ot

    print(data_dict)

    np.save(os.path.join('/media/ders/sdd1/XS/SPCAM_FAMS/data/PubDB/cls_labels.npy'), data_dict) 




if __name__ =="__main__":

    ### function for our private bu image
    # get_K_fold_cross_validation()     # 得到原始五折txt
    # get_split_dataset()    # 得到可训练的五折txt   image id + cls label
    # get_clsnpy()     # 得到 id + cls label 的npy文件

    ### function for public BUSI dataset 
    # convert_BUSI_to_ourfoldertype()
    # get_semmask()
    # get_BUSI_splittxt()
    # get_K_fold_cross_validation()
    # comput_class_balance()
    # get_clsnpy_BUSI()


    ### function for public DatasetB dataset 
    get_DB_splittxt()
    get_K_fold_cross_validation()
    comput_class_balance()
    get_clsnpy_BUSI()

    

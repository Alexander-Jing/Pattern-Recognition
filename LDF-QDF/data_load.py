import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt


class data_load_mnist():
    def __init__(self):
        pass        

    def load_mnist_from_tf(self):
        
        # 导入数据集
        mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

        # 训练集
        train_images = mnist.train.images
        train_labels = mnist.train.labels
        # 验证集
        validation_images = mnist.validation.images
        validation_labels = mnist.validation.labels
        
        # 合并下验证集和训练集
        train_images = np.vstack((train_images,validation_images)) 
        train_labels = np.vstack((train_labels,validation_labels))
        
        # 测试集
        test_images = mnist.test.images
        test_labels = mnist.test.labels
        # mnist 数据集总共是
        # 训练集 60000 * 784
        # 测试集 10000 * 784
            
        print('successfully installed the dataset MNIST')
        return train_images,test_images,train_labels,test_labels
    
    def PCA_all(self,train_images, test_images, yita):  # yita 输入的阈值，用于调整降维的维数
        
        data_all = np.vstack((train_images,test_images))/255  # 合并数据，并且进行归一化，现在的维数是(60000+10000) * 784
        data_cov = (1/data_all.shape[0]) * \
            np.dot((data_all-np.mean(data_all,axis=0)).T,(data_all-np.mean(data_all,axis=0)))  # 计算协方差矩阵，大小为784*784
        
        eigen_value = np.linalg.eigh(data_cov)  # 计算返回特征值和特征向量，注意的是，返回值是[0]是特征值从小到大排列，[1]是特征向量按照特征值的顺序按照列的形式排列
        for i in range(eigen_value[0].shape[0]):  # 如果主成分特征值最大的几个的和占到总的的yita这个阈值
            if ((np.sum(eigen_value[0][eigen_value[0].shape[0]-i : eigen_value[0].shape[0]]) / np.sum(eigen_value[0])) >= yita):
                data_pca = np.dot(data_all, eigen_value[1][:, eigen_value[1].shape[1]-i : eigen_value[1].shape[1]])
                print("successfully make the PCA manipulation")
                print("dimension: %2d"%i)
                print("all data size: " + str(data_pca.shape))
                return data_pca[0:60000, :],data_pca[60000:70000, :]  # 此时返回PCA的值，大小(60000+10000) * k  降至k维
    
    # 提取各个类别的数据
    def get_datasets(self,type_all,train_images,train_labels):  # 这是用来收集训练集中各个类的数据用的
        
        type_lists = []
        for type_i in type_all:
            
            check_type = np.zeros(10)
            check_type[type_i] = 1
            type_list = []
            for i,label in enumerate(train_labels):
                if ((label==check_type).all()):
                    type_list.append(train_images[i])
            type_lists.append(np.array(type_list))  # 将数据提出来，存入列表type_lists中，用于计算
        return type_lists

    # 提取各个类别的数据(训练用)
    def get_datasets_test(self,type_all,test_images,test_labels):  # 这是用来收集训练集中各个类的数据用的
        
        type_lists = []
        type_lists_tests = []
        for type_i in type_all:
            
            check_type = np.zeros(10)
            check_type[type_i] = 1
            type_list = []
            type_lists_test = []
            for i,label in enumerate(test_labels):
                if ((label==check_type).all()):
                    type_list.append(test_images[i])
                    type_lists_test.append(label)
            type_lists.append(np.array(type_list))  # 将数据提出来，存入列表type_lists中，用于计算
            type_lists_tests.append(np.array(type_lists_test)) # 存入对应的数据的label
        return type_lists,type_lists_tests

"""
if __name__ == '__main__':
    data_pre = data_load_mnist()
    _train_images, _test_images, train_labels, test_labels = data_pre.load_mnist_from_tf()
    train_images,test_images = data_pre.PCA_all(_train_images, _test_images,yita=0.98)
    #type_lists = data_pre.get_datasets([0,1,2,3,4,5,6,7,8,9],train_images, train_labels)
    #print([t.shape for t in type_lists])
    type_lists_test, type_lists_labels_test = \
        data_pre.get_datasets_test([0,1,2,3,4,5,6,7,8,9],test_images, test_labels)
    print([t.shape for t in type_lists_test])
    print([t.shape for t in type_lists_labels_test])
"""


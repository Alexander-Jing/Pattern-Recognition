import numpy as np
from data_load import data_load_mnist
import argparse

class LDF():
    def __init__(self,train_images,test_images,train_labels, test_labels, type_lists, type_lists_test, \
        type_lists_labels_test, type_all):
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.type_lists = type_lists
        self.type_lists_test = type_lists_test
        self.type_lists_labels_test = type_lists_labels_test
        print("using the LDF model, try to solve the problem of classification of type "+ str(type_all))
        ldf_pre = self.ldf_model(type_all)
        self.test_model(ldf_pre,type_all)


    def ldf_model(self,type_all):
        
        train_images = self.train_images
        type_lists = self.type_lists
        train_cov = (1/train_images.shape[0]) * \
            np.dot((train_images-np.mean(train_images,axis=0)).T,(train_images-np.mean(train_images,axis=0)))  # 计算协方差矩阵，大小为 k*k (k为降维之后的值) k=260
        train_cov_inv = np.linalg.inv(train_cov)  # 计算逆矩阵260*260
        ldf_pere = []  # 存放ldf每一个类的参数的数组，大小 d*3 d 为想要分类的数目，3为专门用来存储 1.类别数目 2.wi参数 3.w0 参数
        for type_i in type_all:
            type_lists[type_i]
            type_means = np.mean(type_lists[type_i],axis = 0)  # 计算均值 1*260(降维至260维度)
            # 计算ldf的两个参数
            w_i = np.dot(train_cov_inv, type_means.T)
            w_0 = (-1/2)*(np.linalg.multi_dot([type_means.T,train_cov_inv,type_means])+np.log(type_lists[type_i].shape[0]/train_images.shape[0]))
            ldf_pere.append([type_i,w_i,w_0])
        return ldf_pere
    
    def test_model(self,ldf_pere,type_all):

        type_lists_test = self.type_lists_test
        type_lists_labels_test = self.type_lists_labels_test
        # 收集测试数据
        test_data = []
        test_label = []
        test_num = 0
        for type_i in type_all:
            test_data.append(type_lists_test[type_i])
            test_label.append(type_lists_labels_test[type_i])
            test_num += type_lists_test[type_i].shape[0]
        
        # 这一块不知道咋回事，array合体之后没法reshape，以后再修改试试看吧
        """
        test_data = (np.array(test_data)).reshape([test_num,type_lists_test[0].shape[0]])
        test_label = (np.array(test_label)).reshape([test_num,type_lists_labels_test[0].shape[0]])
        """
        # 暂时用这种比较一般的方法
        _test_data = test_data[0]
        _test_label = test_label[0]
        
        for i in range(len(test_data)-1):
            _test_data = np.vstack((_test_data,test_data[i+1]))
            _test_label = np.vstack((_test_label,test_label[i+1]))
        
        # 数据重整为 d*k d*10 (数量乘维数、数量乘10(10类，类的表示方法))
        test_data = _test_data
        test_label = _test_label

        acc_num = 0
        # 测试数据
        for i,data in enumerate(test_data):
            out_max = -np.inf
            type_max = 0
            for model_pere in ldf_pere:
                output = np.dot((model_pere[1]).T,data.T) + model_pere[2]
                if out_max <= output:
                    out_max = output
                    type_max = model_pere[0]
            check_i = np.zeros(10)
            check_i[type_max] = 1
            if ((test_label[i]==check_i).all()) :
                acc_num += 1
        print("test num: %2d, correct num: %2d"%(test_num,acc_num))
        print("accuracy: {:.2%} ".format(acc_num/test_num))
        
        return acc_num/test_num

class QDF():
    def __init__(self,train_images,test_images,train_labels, test_labels, type_lists, type_lists_test, \
        type_lists_labels_test, type_all):
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.type_lists = type_lists
        self.type_lists_test = type_lists_test
        self.type_lists_labels_test = type_lists_labels_test
        print("using the QDF model, try to solve the problem of classification of type "+ str(type_all))
        qdf_pre = self.qdf_model(type_all)
        self.test_model(qdf_pre,type_all)


    def qdf_model(self,type_all):
        
        train_images = self.train_images
        type_lists = self.type_lists
        
        ldf_pere = []  # 存放ldf每一个类的参数的数组，大小 d*3 d 为想要分类的数目，3为专门用来存储 1.类别数目 2.wi参数 3.w0 参数
        for type_i in type_all:
            # 计算每一类对应的协方差矩阵
            train_cov = (1/type_lists[type_i].shape[0]) * \
            np.dot((type_lists[type_i]-np.mean(type_lists[type_i],axis=0)).T,(type_lists[type_i]-np.mean(type_lists[type_i],axis=0)))  # 计算协方差矩阵，大小为 k*k (k为降维之后的值) k=260
            train_cov_inv = np.linalg.inv(train_cov)  # 计算逆矩阵260*260
            
            type_means = np.mean(type_lists[type_i],axis = 0)  # 计算均值 1*260(降维至260维度)
            # 计算ldf的两个参数
            W_i = (-1/2)*(train_cov_inv)
            w_i = np.dot(train_cov_inv, type_means)
            w_0 = (-1/2)*(np.linalg.multi_dot([type_means.T,train_cov_inv,type_means]) + (-1/2)*np.log(np.abs(np.linalg.det(train_cov))+1.0) + np.log(type_lists[type_i].shape[0]/train_images.shape[0]))
            ldf_pere.append([type_i,W_i,w_i,w_0])
        return ldf_pere
    
    def test_model(self,ldf_pere,type_all):

        type_lists_test = self.type_lists_test
        type_lists_labels_test = self.type_lists_labels_test
        # 收集测试数据
        test_data = []
        test_label = []
        test_num = 0
        for type_i in type_all:
            test_data.append(type_lists_test[type_i])
            test_label.append(type_lists_labels_test[type_i])
            test_num += type_lists_test[type_i].shape[0]
        
        # 这一块不知道咋回事，array合体之后没法reshape，以后再修改试试看吧
        """
        test_data = (np.array(test_data)).reshape([test_num,type_lists_test[0].shape[0]])
        test_label = (np.array(test_label)).reshape([test_num,type_lists_labels_test[0].shape[0]])
        """
        # 暂时只能用这种方法了
        _test_data = test_data[0]
        _test_label = test_label[0]
        for i in range(len(test_data)-1):
            _test_data = np.vstack((_test_data,test_data[i+1]))
            _test_label = np.vstack((_test_label,test_label[i+1]))
        
        test_data = _test_data
        test_label = _test_label
        # 数据重整为 d*k d*10 (数量乘维数、数量乘10(10类，类的表示方法))
        
        acc_num = 0
        # 测试数据
        for i,data in enumerate(test_data):
            out_max = -np.inf  # 下面计算生成的output会有小于0的值，所以这里设置成无限小
            type_max = 0
            for model_pere in ldf_pere:
                output = np.linalg.multi_dot([data,model_pere[1],data.T]) + np.dot((model_pere[2]).T,data.T) + model_pere[3]  # 计算qdf的输出
                if out_max <= output:
                    out_max = output
                    type_max = model_pere[0]
            check_i = np.zeros(10)
            check_i[type_max] = 1
            if ((test_label[i]==check_i).all()) :
                acc_num += 1
        print("test num: %2d, correct num: %2d"%(test_num,acc_num))
        print("accuracy: {:.2%} ".format(acc_num/test_num))
        
        return acc_num/test_num

if __name__ == '__main__':
    
    # 命令行使用代码
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m',default="LDF",help="choose the model, LDF or QDF, if input ALL, use both")
    parser.add_argument('--type',"-t",nargs="+", type=int, help="shoose the types you want to classify, eg. 1 0 means you want to classify the types of 1 and 0")
    orders = parser.parse_args()

     # 导入数据
    data_pre = data_load_mnist()
    _train_images, _test_images, train_labels, test_labels = \
        data_pre.load_mnist_from_tf()
    train_images,test_images = \
        data_pre.PCA_all(_train_images, _test_images,yita=0.98)
    type_lists = \
        data_pre.get_datasets([0,1,2,3,4,5,6,7,8,9],train_images, train_labels)
    type_lists_test, type_lists_labels_test = \
        data_pre.get_datasets_test([0,1,2,3,4,5,6,7,8,9],test_images, test_labels)
    
    if str(orders.model) == "QDF":
        qdf_test = QDF(train_images,test_images,train_labels, test_labels, type_lists, type_lists_test, \
        type_lists_labels_test, type_all=orders.type)
    elif str(orders.model) == "LDF":
        ldf_test = LDF(train_images,test_images,train_labels, test_labels, type_lists, type_lists_test, \
        type_lists_labels_test, type_all=orders.type)
    elif str(orders.model) == "ALL":
        ldf_test = LDF(train_images,test_images,train_labels, test_labels, type_lists, type_lists_test, \
        type_lists_labels_test, type_all=orders.type)
        qdf_test = QDF(train_images,test_images,train_labels, test_labels, type_lists, type_lists_test, \
        type_lists_labels_test, type_all=orders.type)


            
            
        


            


     
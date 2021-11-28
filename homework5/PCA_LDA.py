import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio 
import argparse

def PCA(data,dim):
    """PCA
    the Principe Component Analysis for the input data

    args:
        data(array): the input data for PCA, in the form of rows
        dim(int): the dimension for compression 
    
    returns:
        data_pca(array): the data after PCA
    """
    data_cov = (1/data.shape[0]) * \
            np.dot((data-np.mean(data,axis=0)).T,(data-np.mean(data,axis=0)))  # calculate the cov matrix
        
    eigen_value = np.linalg.eigh(data_cov)  # 计算返回特征值和特征向量，注意的是，返回值是[0]是特征值从小到大排列，[1]是特征向量按照特征值的顺序按照列的形式排列
    for i in range(eigen_value[0].shape[0]):  
        if (i==dim):
            yita = np.sum(eigen_value[0][eigen_value[0].shape[0]-i : eigen_value[0].shape[0]]) / np.sum(eigen_value[0])
            data_pca = np.dot(data, eigen_value[1][:, eigen_value[1].shape[1]-i : eigen_value[1].shape[1]])
            print("successfully make the PCA manipulation")
            print("dimension: %2d"%i)
            print("rate: ",yita)  # calculate the rate
            return data_pca
    
def LDA(data_all,data_types,dim):
    """LDA
    the multi-type Linear Discriminate Analysis for the input data

    args:
        data_all(array): the input data for LDA, in the form of rows
        data_types(list) [data of type1(array), data of type2(array), ..., data of typen(array)]: \
            the input data for LDA, in the form of rows, in the form of different types
        dim(int): the dim of the LDA directions, take care  dim <= (num of labels)-1  (https://blog.csdn.net/u013719780/article/details/78312165)
    return:
        data_lda(array): data after LDA
    """
    # calculate the S_t
    S_t = (1/data_all.shape[0]) * \
            np.dot((data_all-np.mean(data_all,axis=0)).T,(data_all-np.mean(data_all,axis=0)))  # calculate the cov matrix

    # calculate the S_w
    S_w = np.zeros(S_t.shape)
    for data_type in data_types:
        S_w += (1/data_type.shape[0]) * \
            np.dot((data_type-np.mean(data_type,axis=0)).T,(data_type-np.mean(data_type,axis=0)))  # calculate the cover matrix for each types
    
    S = np.dot(np.linalg.inv(S_w), S_t-S_w)  # for the eigenvalues and eigenvectors
    
    eigen_value = np.linalg.eigh(S)  # 计算返回特征值和特征向量，注意的是，返回值是[0]是特征值从小到大排列，[1]是特征向量按照特征值的顺序按照列的形式排列
    for i in range(eigen_value[0].shape[0]):  
        if (i==dim):
            data_lda = np.dot(data_all, eigen_value[1][:, eigen_value[1].shape[1]-i : eigen_value[1].shape[1]])
            print("successfully make the LDA manipulation")
            print("dimension: %2d"%i)
            return data_lda

def KNN(data_trains,data_tests,label_train,label_test):
    """KNN model (k=1)
    choose the nearest as the result of the classification

    args:
        data_trains(array): the training set
        data_tests(array): the testing set
        label_train(array 1 column): the training label
        label_test(array 1 column): the testing label
    return:
        types: the classification result
        information(list) [accuracy, ]: the accuracy of the test
    """
    test_pre = []
    for data_test in data_tests:
        dis = (data_test - data_trains)**2
        pre = np.sum(dis,axis=1)  # calculate the distance 
        test_pre.append(label_train[np.argmin(pre)])  # find the minimum andante the classification 
    test_pre = np.array(test_pre)
    
    acc_num = np.count_nonzero(test_pre==label_test)  # .reshape(test_pre.shgape[0],1)
    test_num = test_pre.shape[0]
    print("\n test num: %2d, correct num: %2d"%(test_num,acc_num))
    print("accuracy: {:.2%} \n".format(acc_num/test_num))
    return test_pre, acc_num/test_num


def test(train_data,test_data,data_types,pca_dim,lda_dim):
    """test
    test the PCA+KNN, LDA+KNN on different datasets

    args:
        data_trains(array): the training set
        data_tests(array): the testing set
        data_types(list) [data of type1(array), data of type2(array), ..., data of typen(array)]: \
            the input data for LDA, in the form of rows, in the form of different types
        pca_dim(int): the dimension for compression
        lda_dim(int): the dim of the LDA directions, take care  dim <= (num of labels)-1
    return:
        show the test result 
    """
    # the pca part
    train_all_pca = PCA(np.vstack((train_data[:,0:-1],test_data[:,0:-1])),pca_dim)
    train_data_pca = train_all_pca[0:train_data.shape[0],:]
    test_data_pca = train_all_pca[train_data.shape[0]:,:]
    train_label_pca = train_data[:,-1]
    test_label_pca = test_data[:,-1]
    _,acc_pca = KNN(train_data_pca,test_data_pca,train_label_pca,test_label_pca)
    

    # the lda part
    train_all_lda = LDA(np.vstack((train_data[:,0:-1],test_data[:,0:-1])),data_types,lda_dim)
    train_data_lda = train_all_lda[0:train_data.shape[0],:]
    test_data_lda = train_all_lda[train_data.shape[0]:,:]
    train_label_lda = train_data[:,-1]
    test_label_lda = test_data[:,-1]
    _,acc_lda = KNN(train_data_lda,test_data_lda,train_label_lda,test_label_lda)

    return acc_pca,acc_lda



if __name__ == '__main__':

    # load the training data
    datafile_orl = "ORLData_25.mat"
    ORLData = scio.loadmat(datafile_orl)
    ORLData = (np.array(ORLData['ORLData']).astype(float)).T
    #print(ORLData)
    data_types = []  # form the data_types for lda
    for i in range(1,int(np.max(ORLData[:,-1]))+1):
        ind = np.where(ORLData[:,-1]==float(i))[0].tolist()
        data_types.append(ORLData[ind,:][:,0:-1])
    
    # form the training testing set
    ORL_train = []
    ORl_test = []
    for i in range(1,int(np.max(ORLData[:,-1]))+1):
        ind_all = np.where(ORLData[:,-1]==float(i))[0].tolist()
        ind_train = ind_all[0:8]
        ind_test = ind_all[8:]
        ORL_train.append(ORLData[ind_train,:])
        ORl_test.append(ORLData[ind_test,:])
    ORL_train = np.vstack(ORL_train)
    ORl_test = np.vstack(ORl_test)
    print("the dataset ORLData_25.mat\n")
    acc_pcas = []
    acc_ldas = []
    for i in range(1,51+1):
        acc_pca,acc_lda = test(ORL_train,ORl_test,data_types,5*i,5*i)
        acc_pcas.append(acc_pca)
        acc_ldas.append(acc_lda)
    plt.plot([5*j for j in range(1,len(acc_pcas)+1)],acc_pcas,label="acc_pca")
    plt.plot([5*j for j in range(1,len(acc_ldas)+1)],acc_ldas,label="acc_lda")
    plt.xlabel("dimension")
    plt.ylabel("accuracy")
    plt.title("test result on different dimensions of PCA/LDA ORLData_25")
    plt.legend()
    plt.show()

    datafile_vel = "vehicle.mat"
    VelData = scio.loadmat(datafile_vel)
    VelData = VelData['UCI_entropy_data']
    VelData = VelData['train_data']
    VelData = (VelData[0,0].astype(float)).T
    #print(VelData)
    data_types = []  # form the data_types for lda
    for i in range(1,int(np.max(VelData[:,-1]))+1):
        ind = np.where(VelData[:,-1]==float(i))[0].tolist()
        data_types.append(VelData[ind,:][:,0:-1])
    print("the dataset vehicle.mat\n")
    acc_pcas = []
    acc_ldas = []
    for i in range(1,17+1):
        acc_pca,acc_lda = test(VelData[0:676,:],VelData[677:,:],data_types,i,i)
        acc_pcas.append(acc_pca)
        acc_ldas.append(acc_lda)
    plt.plot([j for j in range(1,len(acc_pcas)+1)],acc_pcas,label="acc_pca")
    plt.plot([j for j in range(1,len(acc_ldas)+1)],acc_ldas,label="acc_lda")
    plt.xlabel("dimension")
    plt.ylabel("accuracy")
    plt.title("test result on different dimensions of PCA/LDA vehicle")
    plt.legend()
    plt.show()



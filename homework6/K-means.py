import numpy as np
import matplotlib.pyplot as plt

def data_form(path):
    """form the data for test and save it
    
    args:
        path(string): save the data 
    return:
        X1: the data in the form of columns
        X: the data in the form of a list
    """
    sigma = [[1,0],[0,1]]  # the convirance matrix
    mus = [[1, -1],[5.5, -4.5],[1, 4],[6, 4.5],[9, 0.0]]  # the means
    X = []
    for mu in mus:
        x = np.random.multivariate_normal(mu,sigma,200)
        X.append(x)
        plt.scatter(x[:,0],x[:,1])
    plt.title("the original points")
    plt.show()
    X1 = np.vstack(X)

    return X1,X

def k_means(mean,X1):
    """the k-means algorithm 
    args:
        mean(array(n,2),n: the cluster types, the point with size (1,2)):\
            the initial mean of the k-means algorithm 
        X(array,size(1000,2)): the input data X for clustering 
    returns:
        X_cluster: the points after clustering 
    """
    X_cluster = []

    while(True): 

        X_cluster = []
        for _ in range(mean.shape[0]):
            _ = []
            X_cluster.append(_)  # set the list for clustering

        for i in range(X1.shape[0]):
            x_attri = X1[i,:]
            dis = mean - x_attri  # calculate the dis between mean and each x
            types = np.argmin(np.sum(dis**2,axis=1))  # know which is the closest type
            
            X_cluster[types].append(x_attri)  # get the type
        
        mean_new = mean.copy()
        for j in range(len(X_cluster)):
            mean_new[j,:] = np.mean(np.vstack(X_cluster[j]),axis=0)  # update the means
        if (mean_new == mean).all():  # if there is no change, stop the iteration 
            break
        else:
            mean = mean_new  # else go on the iterations
    
    return X_cluster,mean
    
def evaluate(X, mean_origin, X_cluster, mean):
    """evaluate the result of the clustering model
    args:
        X(list with n sub list of np.arrays, n types): the original data
        mean_origin(np.array, size(n,2)): the means of the orignial data 
        X_cluster(list with n sub list of np.arrays, n types): the clustering data
        mean(np.array, size(n,2)): the means of the clustering data 
    returns:
        accuracy: the accuracy of the clustering 
        error: the error between the means of the clustering result
    """

    """
    keys = []  # the links from the original data to the clustering data, choose the closest 
    errors = []   # the errors from mean of the original data to the mean of clustering data, choose the smallest 
    mean1 = mean.copy() 
    for i in range(mean_origin.shape[0]):
        _mean = mean_origin[i,:]
        dis = mean1 - _mean  # calculate the dis between mean and each x
        keys.append(np.argmin(np.sum(dis**2,axis=1)))  # know which is the closest type
        errors.append(np.sqrt( np.min( np.sum(dis**2,axis=1) )))  # the smallest error
        mean1 = np.delete(mean1,keys[i],axis=0)
    
    for j in range(len(X)):
        _origin = np.vstack(X[j])
        _cluster = np.vstack(X_cluster[keys[j]])
        cluster_right += np.array([x for x in set(tuple(x) for x in _origin) & set(tuple(x) for x in _cluster)]).shape[0]
    
    accuracy = cluster_right/1000.0
    """
    errors = []   # the errors from mean of the original data to the mean of clustering data, choose the smallest 
    cluster_right = []
    cluster_num = 0
    for i in range(len(X_cluster)):
        cluster_right = []
        _cluster = np.vstack(X_cluster[i])
        for j in range(len(X)):
            _origin = np.vstack(X[j])
            cluster_right.append(np.array([x for x in set(tuple(x) for x in _origin) & set(tuple(x) for x in _cluster)]).shape[0])
        k = np.argmax(np.array(cluster_right))  # find the most accurate cluster
        cluster_num += cluster_right[k]
        dis = mean[i,:] - mean_origin[k,:]
        dis = dis.reshape((1,2))
        errors.append( np.sqrt(np.sum(dis**2,axis=1)) )
    accuracy = cluster_num/1000.0    
    
    return errors,accuracy

    

if __name__ == '__main__':
    X1,X = data_form("")
    print("choose the random mean data as the initial points(group1)")
    mean_origin = np.array([[1.0,-1.0],[5.5,-4.5],[1.0,4.0],[6.0,4.5],[9.0,0.0]])
    mean_ind = np.random.randint(0,X1.shape[0],5)
    _mean = X1[mean_ind,:]
    X_cluster,mean = k_means(_mean,X1)
    for x in X_cluster:
        x = np.vstack(x)
        plt.scatter(x[:,0],x[:,1])
    plt.title("the clustering points")
    plt.show()
    print("centers: ",mean)
    errors,accuracy = evaluate(X, mean_origin, X_cluster, mean)
    print("accuracy: {:.2%} \n".format(accuracy))
    print("errors: ",errors)

    print("choose the random mean data as the initial points(group2)")
    mean_origin = np.array([[1.0,-1.0],[5.5,-4.5],[1.0,4.0],[6.0,4.5],[9.0,0.0]])
    mean_ind = np.random.randint(0,X1.shape[0],5)
    _mean = X1[mean_ind,:]
    X_cluster,mean = k_means(_mean,X1)
    for x in X_cluster:
        x = np.vstack(x)
        plt.scatter(x[:,0],x[:,1])
    plt.title("the clustering points")
    plt.show()
    print("centers: ",mean)
    errors,accuracy = evaluate(X, mean_origin, X_cluster, mean)
    print("accuracy: {:.2%} \n".format(accuracy))
    print("errors: ",errors)

    print("choose the original mean data as the initial points")
    mean_origin = np.array([[1.0,-1.0],[5.5,-4.5],[1.0,4.0],[6.0,4.5],[9.0,0.0]])
    _mean = mean_origin
    X_cluster,mean = k_means(_mean,X1)
    for x in X_cluster:
        x = np.vstack(x)
        plt.scatter(x[:,0],x[:,1])
    plt.title("the clustering points")
    plt.show()
    print("centers: ",mean)
    errors,accuracy = evaluate(X, mean_origin, X_cluster, mean)
    print("accuracy: {:.2%} \n".format(accuracy))
    print("errors: ",errors)
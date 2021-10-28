import numpy as np 




def BatchPerception(x1,x2,a,theta,eta=0.1):
    """batch perception algorithm 

    the batch perception algorithm function based on the PPT

    Args:
        x1: numpy array, the data of the type w1
        X2: numpy array, the data of the type w2
        x1,x2 will be the augmented and normalized after input 

        a: numpy array, default as the zero vector \
            the initial parameter of the model, in accord with the data after augmentation and normalization 
        eta: float number, the iteration step length
        theta: np array, the condition of the end of iteration  

    returns:
        a_iters: list, recording the parameter a in every iteration 
        Y_iters: list , recording the incorrectly classified samples in every iteration

    raises:
        ValueError: the input shouldn't be none
    """

    try:
        if x1.shape[0] <= 1 or x2.shape[0] <=1 :
            raise ValueError("the shape of x1 or x2 is not valid")  # raise the error if the data shape is not enough
    except:
        raise
    
    x1 = np.c_[x1, np.ones(x1.shape[0])]  # the augmentation and normalization of x1
    x2 = (-1)*np.c_[x2, np.ones(x2.shape[0])]  # the augmentation and normalization of x2
    X = np.vstack((x1,x2))  # conbine the x1 x2 together 
    
    k = 0  # record the iteration nums
    Y_iters=[]
    a_iters = []
    while True:
        Y = [x for x in X if np.dot(x,a.T)<=0]  # select the elements that are classified wrong
        Y_iters.append(Y) 
        if len(Y)==0:
            break  # if there are no incorrectly classified samples
        k += 1
        da = eta*(np.array(Y)).sum(axis=0)  
        a += da  # update the parameter a
        a_iters.append(a)
        if (np.abs(da) < theta).all():
            break  # if the modification is small enough, stop the iteration 
    np.set_printoptions(precision=3)
    print("finished")
    print("iterations:",k)
    print('a:',a)
    print('Y:',Y)
    return a_iters,Y_iters

def HoKash(x1,x2,a,b,bmin,eta=0.5,lamda=0.01,kmax=1000):
    """Ho-Kashyap algorithm 

    the Ho-Kashyap algorithm function based on the PPT

    Args:
        x1: numpy array, the data of the type w1
        X2: numpy array, the data of the type w2
        x1,x2 will be the augmented and normalized after input 

        a: numpy array, default as the zero vector \
            the initial parameter of the model, in accord with the data after augmentation and normalization 
        b: numpy array, default as the ones vector 

        eta: float number, the iteration step length
        lamda: float number, the parameter for the matrix inverse 
        kmax: largest iteration times
        bmin: np array, the condition of the end of iteration  

    returns:
        a_iters: list, recording the parameter a in every iteration 
        b_iters: list, recording the parameter a in every iteration
    raises:
        ValueError: the input shouldn't be none
    """
    try:
        if x1.shape[0] <= 1 or x2.shape[0] <=1 :
            raise ValueError("the shape of x1 or x2 is not valid")  # raise the error if the data shape is not enough
    except:
        raise
    
    x1 = np.c_[x1, np.ones(x1.shape[0])]  # the augmentation and normalization of x1
    x2 = (-1)*np.c_[x2, np.ones(x2.shape[0])]  # the augmentation and normalization of x2
    Y = np.vstack((x1,x2))  # conbine the x1 x2 together as the matrix in Y PPT 

    k = 0
    a_iters = []
    b_iters = []
    a_iters.append(a)
    b_iters.append(b)
    while k<kmax:
        k += 1
        e = np.dot(Y,a) - b
        e = (1/2)*(e + np.abs(e))  # the important part in updating b
        b += 2*eta*e  # update b
        a = np.linalg.multi_dot([np.linalg.inv(np.dot(Y.T,Y)+lamda*np.eye(np.dot(Y.T,Y).shape[0])), Y.T, b])  # update a
        a_iters.append(a)
        b_iters.append(b)  # record the iteration 

        if (np.abs(e)<=bmin).all():
            break
    np.set_printoptions(precision=3)
    print("finished")
    print("iterations:",k)
    print("a.T:",a.T)
    print("b.T:",b.T)
    print("errors:",e.T)
    return a_iters,b_iters

def MSE_multiType(X_train,Y_train,X_test,Y_test,lamda=0.01):
    """MSE multi types

    the MSE multi algorithm according to the PPT 

    Args:
        X_train: np array, the input training data,should be augmented
        Y_train: np array, the classification types of the training data 
        X_test: np array, the input testing data,should be augmented
        Y_test: np array, the classification types of the testing data 
        lamda: float number, the parameter for matrix calculation 

    returns:
        W: np array, the paramter for classification 

    raises:
        none 
    """
    X_train = np.c_[X_train, np.ones(X_train.shape[0])].T  # the augmentation of X
    X_test = np.c_[X_test, np.ones(X_test.shape[0])].T
    W = np.linalg.multi_dot([np.linalg.inv(np.dot(X_train,X_train.T)+lamda*np.eye(np.dot(X_train,X_train.T).shape[0])), X_train, Y_train.T])  # calculate the W parameter 
    
    Y_pre = np.dot(W.T,X_test)
    Y_pre = Y_pre.T
    for i in range((Y_pre).shape[0]):  # take care of the form of prediction 
        y = np.zeros(Y_pre.shape[1])
        y[np.argmax(Y_pre[i])] = 1
        Y_pre[i] = y  # form the output of the prediction 
    Y_test = Y_test.T
    k = 0
    for i in range((Y_pre).shape[0]):  # check the accuracy 
        if (Y_pre[i] == Y_test[i]).all():
            k += 1
    np.set_printoptions(precision=3)
    print("test set:",Y_test)
    print("predict:",Y_pre)
    print("accuracy:", k/(Y_pre).shape[0])
    return W

            
if __name__=="__main__":
    # the data set
    x_w1 = np.array([[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8], [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2],
                [5.0, 6.4], [3.9, 4.0]])   # w1
    x_w2 = np.array([[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9], [1.4, -3.2], [2.4, -4.0], [2.5, -6.1],
                [8.4, 3.7], [4.1, -2.2]])  # w2
    x_w3 = np.array([[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2], [-1.3, 3.7], [-3.4, 6.2],
                [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]])  # w3  
    x_w4 = np.array([[-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0], [-0.5, -9.2], [-5.3, -6.7],
                [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]])  # w4
    # batch perception implementations on the w1,w2 and on the w3,w4
    print("BatchPerception")
    BatchPerception(x_w1,x_w2,a = np.zeros(x_w1.shape[1]+1),theta=np.array([0.01 for _ in range(x_w1.shape[1]+1)])) 
    BatchPerception(x_w3,x_w2,a = np.zeros(x_w3.shape[1]+1),theta=np.array([0.01 for _ in range(x_w1.shape[1]+1)]))
    # HoKash algorithm implementation  
    print("HoKash")
    HoKash(x_w1,x_w3,a=np.ones((x_w1.shape[1]+1,1)), b=np.ones((x_w1.shape[0]+x_w3.shape[0],1)),\
        bmin=np.array([0.05 for _ in range(x_w1.shape[0]+x_w3.shape[0])]).reshape((x_w1.shape[0]+x_w3.shape[0],1)), eta=0.05)
    HoKash(x_w2,x_w4,a=np.ones((x_w2.shape[1]+1,1)), b=np.ones((x_w2.shape[0]+x_w4.shape[0],1)),\
        bmin=np.array([0.05 for _ in range(x_w2.shape[0]+x_w4.shape[0])]).reshape((x_w2.shape[0]+x_w4.shape[0],1)), eta=0.05)
    
    print("MSE_multiType")
    # test the MSE multi types algorithm 
    X_train = np.vstack((x_w1[0:8,:],x_w2[0:8,:],x_w3[0:8,:],x_w4[0:8,:]))  # form the training set
    X_test = np.vstack((x_w1[8:,:],x_w2[8:,:],x_w3[8:,:],x_w4[8:,:]))
    y1 = np.array([[1,0,0,0] for _ in range(8)])
    y2 = np.array([[0,1,0,0] for _ in range(8)])
    y3 = np.array([[0,0,1,0] for _ in range(8)])
    y4 = np.array([[0,0,0,1] for _ in range(8)])
    Y_train = np.vstack((y1,y2,y3,y4)).T 

    y1 = np.array([[1,0,0,0] for _ in range(2)])
    y2 = np.array([[0,1,0,0] for _ in range(2)])
    y3 = np.array([[0,0,1,0] for _ in range(2)])
    y4 = np.array([[0,0,0,1] for _ in range(2)])
    Y_test = np.vstack((y1,y2,y3,y4)).T 
    MSE_multiType(X_train,Y_train,X_test,Y_test)
    
    
    

        



    

    
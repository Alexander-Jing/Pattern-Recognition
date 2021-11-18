import network
from network import node, NetStructure
import matplotlib.pyplot as plt

#第一类 10 个样本（三维空间）：
a = [[ 1.58, 2.32, -5.8], [ 0.67, 1.58, -4.78], [ 1.04, 1.01, -3.63], 
[-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
[ 1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [ 0.45, 1.33, -4.38],
[-0.76, 0.84, -1.96]]
 
#第二类 10 个样本（三维空间）：
b = [[ 0.21, 0.03, -2.21], [ 0.37, 0.28, -1.8], [ 0.18, 1.22, 0.16], 
[-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
[-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [ 0.44, 1.31, -0.14],
[ 0.46, 1.49, 0.68]]
#第三类 10 个样本（三维空间）：
c = [[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [ 1.55, 0.99, 2.69], 
[1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
[1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [ 0.25, 0.68, -0.99],
[ 0.66, -0.45, 0.08]]

if __name__ == '__main__':
    # form the train data set, because of the lack of the dataset, we don't form the dev set and test set
    train_data = a+b+c
    train_label = [[1,0,0] for _ in range(10)] + [[0,1,0] for _ in range(10)] + [[0,0,1] for _ in range(10)]
    #print(train_label)
    
    HM4net = NetStructure(input=3,hidden=8,output=3)
    losses = HM4net.train(train_data,train_label,epoch=500,rate=0.1)
    plt.plot([i for i in range(1,len(losses)+1)],losses)
    plt.title("Using the SGD optimization,learning rate=0.1,num of nodes in hidden layers: 8")
    plt.xlabel("epoches")
    plt.ylabel("training loss (MSE)")
    plt.show()
    
    HM4net = NetStructure(input=3,hidden=8,output=3)
    losses = HM4net.train_batch(train_data,train_label,epoch=500,rate=0.1)
    plt.plot([i for i in range(1,len(losses)+1)],losses)
    plt.title("Using the mini-batch optimization,learning rate=0.1,num of nodes in hidden layers: 8")
    plt.xlabel("epoches")
    plt.ylabel("training loss (MSE)")
    plt.show()
    
    
    
    HM4net = NetStructure(input=3,hidden=4,output=3)
    losses = HM4net.train_batch(train_data,train_label,batch_size=15,epoch=500,rate=0.1)
    plt.plot([i for i in range(1,len(losses)+1)],losses,label='4 nodes in hidden layers')
    
    HM4net1 = NetStructure(input=3,hidden=8,output=3)
    losses1 = HM4net1.train_batch(train_data,train_label,batch_size=15,epoch=500,rate=0.1)
    plt.plot([i for i in range(1,len(losses1)+1)],losses1,label='8 nodes in hidden layers')
    
    HM4net2 = NetStructure(input=3,hidden=16,output=3)
    losses2 = HM4net2.train_batch(train_data,train_label,batch_size=15,epoch=500,rate=0.1)
    plt.plot([i for i in range(1,len(losses2)+1)],losses2,label='16 nodes in hidden layers')

    HM4net3 = NetStructure(input=3,hidden=32,output=3)
    losses3 = HM4net3.train_batch(train_data,train_label,batch_size=15,epoch=500,rate=0.1)
    plt.plot([i for i in range(1,len(losses3)+1)],losses3,label='32 nodes in hidden layers')

    plt.legend()
    plt.title("Different num of nodes in hidden layers (learning rate=0.1, mini-batch)")
    plt.xlabel("epoches")
    plt.ylabel("training loss (MSE)")
    plt.show()
    



    HM4net = NetStructure(input=3,hidden=8,output=3)
    losses = HM4net.train_batch(train_data,train_label,batch_size=15,epoch=500,rate=0.001)
    plt.plot([i for i in range(1,len(losses)+1)],losses,label='rate=0.001')
    
    HM4net1 = NetStructure(input=3,hidden=8,output=3)
    losses1 = HM4net1.train_batch(train_data,train_label,batch_size=15,epoch=500,rate=0.01)
    plt.plot([i for i in range(1,len(losses1)+1)],losses1,label='rate=0.01')
    
    HM4net2 = NetStructure(input=3,hidden=8,output=3)
    losses2 = HM4net2.train_batch(train_data,train_label,batch_size=15,epoch=500,rate=0.1)
    plt.plot([i for i in range(1,len(losses2)+1)],losses2,label='rate=0.1')

    HM4net3 = NetStructure(input=3,hidden=8,output=3)
    losses3 = HM4net3.train_batch(train_data,train_label,batch_size=15,epoch=500,rate=1.0)
    plt.plot([i for i in range(1,len(losses3)+1)],losses3,label='rate=1.0')

    plt.legend()
    plt.title("Different learning rates (8 nodes in hidden layers, mini-batch)")
    plt.xlabel("epoches")
    plt.ylabel("training loss (MSE)")
    plt.show()

import numpy
import numpy as np
# import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd

"""
alpha、tau、K、DC、init、tol 六个输入参数的无严格要求；
alpha 带宽限制 经验取值为 抽样点长度 1.5-2.0 倍；
tau 噪声容限 ；
K 分解模态（IMF）个数；
DC 合成信号若无常量，取值为 0；若含常量，则其取值为 1；
init 初始化 w 值，当初始化为 1 时，均匀分布产生的随机数；
tol 控制误差大小常量，决定精度与迭代次数
"""
datasets = 'SSE'


data_addr = '../data/data/'+datasets+'.npy'
canshu = pd.read_csv('../result/'+datasets+'_GA.csv', header=None).values
canshu = canshu.astype(int)
tdata = np.load(data_addr)
print(tdata.shape)
tau = 0.  # noise-tolerance (no strict fidelity enforcement)
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7

for j in range(tdata.shape[0]):
    data = tdata[j]
    K = canshu[j][0]
    alpha = canshu[j][1]

    row = data.shape[0]
    train_size = int(row * 0.8)
    train_data = data[0:train_size]
    test_data = data[train_size:row]
    u2 = []
    # train_data = np.reshape(train_data, [train_data.shape[1], train_data.shape[0]])
    # test_data = np.reshape(test_data, [test_data.shape[1], test_data.shape[0]])
    for i in range(data.shape[1]):
        u, u_hat, omega = VMD(train_data[:, i], alpha, tau, K, DC, init, tol)
        u1, u_hat1, omega1 = VMD(test_data[:, i], alpha, tau, K, DC, init, tol)
        u = list(map(list, zip(*u)))
        u1 = list(map(list, zip(*u1)))
        u2.append(u + u1)

    u2 = np.array(list(map(list, zip(*u2))))
    data = data.astype(float)
    # res = data - np.sum(u2, axis=2)
    print(train_data.shape, np.array(u).shape)
    print(test_data.shape, np.array(u1).shape)
    print(data.shape, u2.shape)
    for i in range(K):
        numpy.savetxt(
            "../data/data/VMDdata/" + datasets + "/" + str(j)+ str(K) + "-" + str(i + 1) + ".csv",
            u2[:, :, i], delimiter=",")
print("Finish\n")




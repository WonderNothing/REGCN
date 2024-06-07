from __future__ import print_function, division
import csv
import pandas as pd
import numpy as np
import glob,os
# tensorflow 1.15.0 + python 3.7
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, GRUCell, CuDNNLSTM, BatchNormalization, RNN, TimeDistributed
from tensorflow.keras.layers import Dense, RNN, TimeDistributed
from input_data import preprocess_data, load_price_data,data_y
from utils import get_trend, avg_relative_error, get_vague_trend, calculate_laplacian
from dgcgru import gcgru
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
from configparser import ConfigParser
import matplotlib.pyplot as plt


config_file_addr = "config.ini"
config = ConfigParser()
config.read(config_file_addr)
data_addr = config["hyper"]["data_addr"]
adj_addr = config["hyper"]["adj_addr"]
adj2_addr = config["hyper"]["adj_type2"]
s_index = int(config["hyper"]["s_index"])
lr = float(config["hyper"]["lr"])
n_neurons = int(config["hyper"]["n_neurons"])
seq_len = int(config["hyper"]["seq_len"])
n_epochs = int(config["hyper"]["n_epochs"])
batch_size = int(config["hyper"]["batch_size"])
n_off = int(config["hyper"]["n_off"])
all_data = int(config["hyper"]["all_data"])
start_index = int(config["hyper"]["start_index"])
VMD_addr =  config["hyper"]["VMD_addr"]
datasets = config["hyper"]["datasets"]


data_addr = data_addr+datasets+'.npy'
data = np.load(data_addr,allow_pickle=True)

r_mse = float(config["hyper"]["r_mse"])
r_acc = float(config["hyper"]["r_acc"])

def unautoNorm(data,mins,maxs): #传入一个矩阵


    ranges = maxs - mins #最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data)) #生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0] #返回 data矩阵的行数
    normData = data * np.tile(ranges,1) #data矩阵每一列数据都除去每一列的差值(差值 = 某列的最大值- 某列最小值)
    normData = normData + np.tile(mins,1) #data矩阵每一列数据都减去每一列的最小值
    return normData

def trainmodel(tdata, tadj, s_index, lr, n_neurons,
         seq_len, n_epochs,j):

    data = tdata.astype(float)
    adj = tadj.astype(float)
    labels = data[:, 3]
    train_rate = 0.8
    pre_len = 1
    time_len = data.shape[0]
    n_gcn_nodes = data.shape[1]

    X_train, y_train, X_test, y_test, pre_y_test = preprocess_data(
        data, labels, time_len, train_rate, seq_len, pre_len)
    y_train = np.expand_dims(y_train, -1)
    p = tf.sparse.to_dense(calculate_laplacian(adj[0]), default_value=0)
    sp = tf.sparse.to_dense(calculate_laplacian(adj[1]), default_value=0)
    DTW = tf.sparse.to_dense(calculate_laplacian(adj[2]), default_value=0)
    Madj = tf.stack([p,sp,DTW], axis=0)

    cell = gcgru(n_neurons, Madj, n_gcn_nodes, 3)

    model = Sequential()


    Ge.fit(
        x=X_train,
        y=y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=1,
        # steps_per_epoch = X_train.shape[0] // batch_size,
        callbacks=[lr_scheduler]
        # validation_data=([X_val, adj2], y_val)
    )
    model_weights_path = './model/model_'+datasets+'-'+str(s_index)+'-weights.h5'  # Path to save the model's parameters
    Ge.save_weights(model_weights_path)
    result = Ge.predict(X_test, batch_size=batch_size,verbose=0)
    return result

def main(data, s_index, lr, n_neurons,
         seq_len, n_epochs):
    # hyperperameter
    adj_addr1 = adj_addr + datasets + '/' + datasets + '_VMD_'+str(s_index)+ adj2_addr
    adj = np.load(adj_addr1, allow_pickle=True)

    tdata = data[s_index]
    tdata = tdata.astype(float)
    labels = tdata[:, 3]
    train_rate = 0.8
    pre_len = 1
    time_len = tdata.shape[0]
    # print(time_len)
    y_test, pre_y_test = data_y(labels, time_len, train_rate, seq_len, pre_len)
    y_test = np.expand_dims(y_test, -1)
    file = glob.glob(os.path.join("%s%s/%s_*.csv" % (VMD_addr, datasets, s_index)))
    VMD = []
    for f in file:
        VMD.append(pd.read_csv(f, header=None).values)
    # num = len(VMD)
    # print(num)
    result = []
    j = 0
    for ndata in (VMD):
        mdata = ndata[0:time_len]
        result1 = trainmodel(mdata, adj[j], s_index, lr, n_neurons,
                             seq_len, n_epochs,j)
        j += 1
        mins = ndata[time_len][3]
        maxs = ndata[time_len + 1][3]
        result.append(unautoNorm(result1, mins, maxs))
        # result.append(result1)

    result = np.sum(result, axis=0)
    print(result.shape)
    result = result[:, -1]
    y_test = y_test[:, -1]

    actual_trend = get_trend(pre_y_test, y_test)
    predicted_trend = get_trend(pre_y_test, result)
    accuracy = accuracy_score(actual_trend, predicted_trend)

    print("***********************")
    print(j)
    print("accuracy: ", accuracy)
    # print("accuracy: ", accuracy1)
    r2 = r2_score(y_test, result)
    print("r2: ", r2)
    rmse = sqrt(mean_squared_error(y_test, result))
    print("rmse: ", rmse)
    mae = mean_absolute_error(y_test, result)
    print("mae: ", mae)
    re = avg_relative_error(y_test, result)
    print("re: ", re)
    write_data = ["REGCN_"+str(seq_len), str(s_index),str(accuracy), str(r2), str(rmse), str(mae), str(re),str(r_mse),str(r_acc)]
    with open('../result/'+datasets+'/result_REGCN.csv', 'a',newline='', encoding='UTF8') as f:
        d = csv.writer(f)
        d.writerow(write_data)

    if all_data != 1:
        plt.plot(y_test, color='red', label='Real Stock Price')
        plt.plot(result, color='blue', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(r'../result/REGCN_'+datasets+'-'+str(s_index)+'.png', dpi=200)
        plt.show()



if __name__ == '__main__':
    if all_data == 1:

        for s_index in range(start_index, data.shape[0]):
            main(data,
                 s_index, lr, n_neurons, seq_len, n_epochs)
    else:
        main(data,
             s_index, lr, n_neurons, seq_len, n_epochs)
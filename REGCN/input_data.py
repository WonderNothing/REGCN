import numpy as np
import pandas as pd
import pickle as pkl


def load_price_data(data_addr, adj_addr):
    adj = pd.read_csv(adj_addr, header=None).values
    price = pd.read_csv(data_addr, header=None).values
    return price, adj
def load_VMD_price_data(VMD31,VMD32,VMD33,data_addr, adj_addr):
    adj = pd.read_csv(adj_addr, header=None).values
    data = pd.read_csv(data_addr, header=None).values
    VMD31 = pd.read_csv(VMD31, header=None).values
    VMD32 = pd.read_csv(VMD32, header=None).values
    VMD33 = pd.read_csv(VMD33, header=None).values
    return VMD31,VMD32,VMD33,data, adj

def preprocess_data(data, labels, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]
    train_label = labels[0:train_size]
    test_label = labels[train_size:time_len]
    pre_test_label = labels[train_size-1:time_len-1]

    trainX, trainY, testX, testY, pre_testY = [], [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len + 1):
        a = train_data[i: i + seq_len + pre_len]
        b = train_label[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])  # seq_len 12
        trainY.append(b[pre_len: seq_len + pre_len])  # pre_len 1
    for i in range(len(test_data) - seq_len - pre_len + 1):
        a = test_data[i: i + seq_len + pre_len]
        b = test_label[i: i + seq_len + pre_len]
        c = pre_test_label[i: i + seq_len + pre_len]
        testX.append(a[0: seq_len])
        testY.append(b[pre_len: seq_len + pre_len])
        pre_testY.append(c[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    pre_testY1 = np.array(pre_testY)

    return trainX1, trainY1, testX1, testY1, pre_testY1
def data_y(labels, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    test_label = labels[train_size:time_len]
    pre_test_label = labels[train_size-1:time_len-1]

    testY, pre_testY = [], []
    for i in range(len(test_label) - seq_len - pre_len + 1):
        b = test_label[i: i + seq_len + pre_len]
        c = pre_test_label[i: i + seq_len + pre_len]
        testY.append(b[pre_len: seq_len + pre_len])
        pre_testY.append(c[seq_len: seq_len + pre_len])
    testY1 = np.array(testY)
    pre_testY1 = np.array(pre_testY)
    return  testY1, pre_testY1
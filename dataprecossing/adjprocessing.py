import glob,os
import numpy as np
from fastdtw import fastdtw
import pandas as pd
from scipy.spatial.distance import euclidean

datasets = 'SSE'
data_addr = '../data/data/'
data = np.load(data_addr+datasets+'_fea.npy',allow_pickle=True)
threshold = 0.9
for j in range(data.shape[0]):
    adj = []
    file = glob.glob(os.path.join("../data/data/VMDdata/%s/%s*.csv" % (datasets, j)))
    for f in file:
        tdata = pd.read_csv(f, header=None).values
        pf = pd.DataFrame(tdata)
        na = pf.corr(method='spearman')
        nb = pf.corr(method='pearson')
        na = np.array(na)
        row, col = np.diag_indices_from(na)
        na[row, col] = 0
        nb = np.array(nb)
        row, col = np.diag_indices_from(nb)
        nb[row, col] = 0
        ndtw_data = np.zeros(shape=(tdata.shape[1], tdata.shape[1]))
        for m in range(tdata.shape[1]):
            for n in range(m + 1, tdata.shape[1]):
                df = tdata
                p, path1 = fastdtw(df[:, m].reshape(-1, 1), df[:, n].reshape(-1, 1), dist=euclidean)
                d = 1 - p / 10 * tdata.shape[0]
                ndtw_data[m][n] = d
                ndtw_data[n][m] = d
        adj.append( [nb, na, ndtw_data])
    print(np.array(adj).shape)
    np.save('../data/adj/' + datasets + '/' + datasets + '_VMD_'+str(j)+'.npy', adj)
    adj = np.array(adj)
    adj[adj < threshold] = 0
    np.save('../data/adj/' + datasets + '/' + datasets + '_VMD_' + str(j) + '_90.npy', adj)

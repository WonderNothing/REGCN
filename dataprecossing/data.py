import glob
import numpy as np
import pandas as pd

import os

datasets = 'SSE'

dtw_addr = '../data/data/raw data/'+datasets+'/'
file = glob.glob(os.path.join('%s*.csv') % (dtw_addr))
result = []
n = 0
for f in file:
    filename, extension = os.path.splitext(f)
    idata = pd.read_csv(f, header=None).values
    if datasets=='DJIA':
        data = idata[1:-1, 1:].astype(float)
    else:
        data = idata[1:-1, 3:].astype(float)
    result.append(data)
    n += 1
print(np.array(result).shape)
np.save('../data/data/' + datasets + '.npy', result)

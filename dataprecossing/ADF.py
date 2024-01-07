from statsmodels.tsa.stattools import adfuller
import pandas as pd

data = pd.read_csv('../data/data/50sh.csv', header=None, sep=',', dtype=float)
tdata = data.iloc[:, 0].values
result = adfuller(tdata)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
for i in range(1,4):
    VMDdata = pd.read_csv('../data/VMDdata_without precessing/GA_50sh/50sh_person_3-'+str(i)+'.csv', header=None, sep=',', dtype=float)
    # data为时间序列数据
    tdata = VMDdata.iloc[:, 0].values
    result = adfuller(tdata)
    print(i)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
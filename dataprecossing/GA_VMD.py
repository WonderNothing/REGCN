import csv

from sklearn import svm
import numpy as np
from numpy.random import random as rand
import random
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vmdpy import VMD
from scipy.fftpack import hilbert, fft, ifft
from math import log
import pandas as pd
import copy
# from tensorflow.keras.losses import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_squared_error

datasets = 'SSE'
data_addr = '../data/data/'+datasets+'.npy'

data = np.load(data_addr,allow_pickle=True)
data = data.astype(float)
print(data.shape)

tau = 0.  # noise-tolerance (no strict fidelity enforcement)
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7

def Fun( x,data1):
    K = int(x[0])
    alpha = int(x[1])

    if K <= low[0]:
        K = low[0]
    if K >= ub[0]:
        K = ub[0]

    if alpha <= low[1]:
        alpha = low[1]
    if alpha >= ub[1]:
        alpha = ub[1]
    s = 0
    for i in range(data.shape[2]):
        u, u_hat, omega = VMD(data1[:,i], alpha, tau, K, DC, init, tol)
        u1 = np.sum(u, axis=0)
        u2 = list(map(lambda x: x[0] - x[1], zip(data1[:,i], u1)))

        u3 = list(map(list, zip(*[data1[:,i],u2])))
        df = pd.DataFrame(u3)
        s += df.corr()[0][1]
    s /= data.shape[1]

    return s

class GAIndividual:
    '''

    创建pop中的单个个体
    '''

    def __init__(self, vardim, bound,x1):
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0
        self.x1 = x1

    def generate(self):
        '''
        generate a random chromsome for genetic algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0][i] + \
                            (self.bound[1][i] - self.bound[0][i]) * rnd[i]
        print(int(self.chrom[0]), self.chrom[1])

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = Fun( self.chrom,self.x1)
        # print(self.fitness)

class GeneticAlgorithm:

    def __init__(self, sizepop, vardim, bound, MAXGEN, params,x1):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        param: algorithm required parameters, it is a list which is consisting
               of crossover rate, mutation rate, alpha
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params
        self.x1 = x1
        # self.i = i
    def initialize(self):

        for i in range(0, self.sizepop):
            ind = GAIndividual(self.vardim, self.bound,self.x1)
            ind.generate()
            self.population.append(ind)

    def evaluate(self):

        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def solve(self):

        self.t = 0  # 迭代次数
        self.initialize()  # 初始化种群
        self.evaluate()  # 计算适应度
        best = np.min(self.fitness)  # 选出适应度最小的个体
        bestIndex = np.argmin(self.fitness)  # 最小适应度的索引
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)  # 平均适应度
        self.BEST = []
        while (self.t < self.MAXGEN):
            print('迭代次数：', self.t)
            self.t += 1
            self.selectionOperation()  # 选择
            self.crossoverOperation()  # 交叉
            self.mutationOperation()  # 变异
            self.evaluate()  # 重新计算新种群适应度
            best = np.min(self.fitness)
            bestIndex = np.argmin(self.fitness)
            if best < self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.BEST.append(self.best)

        print("Optimal solution is:", int(self.best.chrom[0]), int(self.best.chrom[1]))
        result = [int(self.best.chrom[0]), int(self.best.chrom[1])]
        with open('../result/' + datasets + '_GA.csv', 'a',newline='', encoding='UTF8',) as f:
            d = csv.writer(f)
            d.writerow(result)


    def selectionOperation(self):
        '''
        selection operation for Genetic Algorithm
        '''
        newpop = []
        totalFitness = np.sum(self.fitness)
        accuFitness = np.zeros((self.sizepop, 1))

        # 适应度的累进占比
        sum1 = 0.
        for i in range(0, self.sizepop):
            accuFitness[i] = sum1 + self.fitness[i] / totalFitness
            sum1 = accuFitness[i]

        # 随机选出新种群的索引
        for i in range(0, self.sizepop):
            r = random.random()
            idx = 0
            for j in range(0, self.sizepop - 1):
                if j == 0 and r < accuFitness[j]:
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[idx])
        self.population = newpop

    def crossoverOperation(self):
        '''
        crossover operation for genetic algorithm
        '''
        newpop = []
        # 选出两个个体进行交换
        for i in range(0, self.sizepop, 2):
            idx1 = random.randint(0, self.sizepop - 1)
            idx2 = random.randint(0, self.sizepop - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = random.random()

            if r < self.params[0]:
                crossPos = random.randint(1, self.vardim - 1)
                for j in range(crossPos, self.vardim):
                    newpop[i].chrom[j] = newpop[i].chrom[j] * self.params[2] + \
                                         (1 - self.params[2]) * newpop[i + 1].chrom[j]

                    newpop[i + 1].chrom[j] = newpop[i + 1].chrom[j] * self.params[2] + \
                                             (1 - self.params[2]) * newpop[i].chrom[j]
        self.population = newpop

    def mutationOperation(self):
        '''
        mutation operation for genetic algorithm
        '''
        newpop = []
        for i in range(0, self.sizepop):
            newpop.append(copy.deepcopy(self.population[i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:

                    newpop[i].chrom[mutatePos] = newpop[i].chrom[mutatePos] - \
                                                 (newpop[i].chrom[mutatePos] - self.bound[0][mutatePos]) * \
                                                 (1 - random.random() ** (1 - self.t / self.MAXGEN))
                else:

                    newpop[i].chrom[mutatePos] = newpop[i].chrom[mutatePos] + \
                                                 (self.bound[1][mutatePos] - newpop[i].chrom[mutatePos]) * \
                                                 (1 - random.random() ** (1 - self.t / self.MAXGEN))
        self.population = newpop

if __name__ == "__main__":
    for i in range (data.shape[0]):
        tdata = data[i]
        train_size = int(tdata.shape[0] * 0.8)
        # print(train_size)
        train_data = tdata[0:train_size]
        x1 = train_data
        low = [2, x1.shape[0]/2]
        ub = [5, x1.shape[0]*3]
        bound = [low, ub]

        # def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        # sizepop：总体sizepop
        # vardim：变量的维度
        # 绑定：变量的边界
        # MAXGEN：终止条件
        # param：算法所需参数，它是一个列表，由[交叉率、突变率、α]
        ga = GeneticAlgorithm(60, 2, bound, 100, [0.9, 0.1, 0.5],x1)
        ga.solve()

#
#
#




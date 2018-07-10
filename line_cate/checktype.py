import os,sys
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Series
sys.path.append('../data_produce')
from type1 import *
from type2 import *
from type3 import *
from type4 import *

ts=type1()
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(ts)
plt.plot(ts)
plt.show()
ts_diff=ts.diff(periods=1)
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(ts_diff)
plt.plot(ts_diff)
plt.show()
four=ts_diff.describe()
print(four)
Q1 = four['25%']
Q3 = four['75%']
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR
print('upper:%s',upper)
print('lower:%s',lower)

exv={}
for i,v in ts_diff.items():
    if(v<lower):
        exv[i-1]=v
exv[ts_diff.size-1]=ts_diff[ts_diff.size-1]
print(exv)

prel=-1
preh=0
temp=0
it=0
for i,v in exv.items():
    tempp=i-it
    if(tempp>temp):
        temp=tempp
        prel=preh+1
        preh=i
    it=i+1
print('the start and end is %s%s:',prel,preh)


ts_a=ts[prel:preh+1]
plt.plot(ts_a)
plt.show()


#return ndarray
y=ts_a.values
x=ts_a.index.values

import numpy as np
from scipy.optimize import leastsq
def fun(p, x):
    """
    定义想要拟合的函数
    """
    k, b = p  #从参数p获得拟合的参数
    return k*x+b

def err(p, x, y):
    """
    定义误差函数
    """
    return fun(p,x) -y

p0 = [1,1]

param = leastsq(err, p0, args=(x,y))
print(param[0])
k,b=param[0]
print('k,b:%s %s',k,b)
end=ts.index[ts.size-1]
print(end)
pred_x=[]
for i in range(20):
    end=end+1
    pred_x.append(end)

pred_x=np.array(pred_x)
pred_y=k*pred_x+b
dd=pred_y[0]-ts[ts.size-1]

print(pred_x)
print(pred_y)
pred_ts=Series(pred_y-dd)
pred_ts.index=pred_x
print(pred_ts)
plt.plot(ts)
plt.plot(pred_ts)
plt.show()

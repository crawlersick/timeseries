#导入需要的模块
import csv
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from scipy import  stats

#从csv文件中读取时间和对应的数据
data = pd.read_csv("test.csv")

time_series = []
mydata = []
#将时间保存在time列表中，而将数据保存在mydata中
#将.csv数据读入Pandas数据帧
for timestamp,used_percent in zip(data['t'],data['d']):
    #将数字时间转换为能够识别的时间 
    time_series.append(timestamp)
    mydata.append(used_percent)

dta=pd.Series(mydata)
dta.index = pd.Index(time_series)
dta.index = pd.DatetimeIndex(dta.index)
#dta = dta.resample('H')
#dta = dta.fillna(np.mean(mydata))

dta.plot()
plt.show()
#print(dta)

dta= dta.diff(1)#我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta,lags=40,ax=ax2)
plt.show()

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
from statsmodels.tsa.arima_model import ARIMA

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

#model = ARIMA(dta, (p,d,q), freq='H').fit(trend='nc')
#从之前的分析，我们定下p=0,d=1,q=0
model = ARIMA(dta, (0,1,0),freq='H').fit(trend='nc')
#predict_outcome中保存了预测结果、标准误差以及置信区间
predict_outcome = model.forecast(24)
#我们可以从predict_outcome[0]获取到预测的结果
print (predict_outcome[0])
fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.plot()
predict_outcome.plot(ax=ax)

plt.show()


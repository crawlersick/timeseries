#导入需要的模块
import csv
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
#从csv文件中读取时间和对应的数据
data = pd.read_csv("test.csv")

time_series = []
mydata = []
#将时间保存在time列表中，而将数据保存在mydata中
#将.csv数据读入Pandas数据帧
dt = datetime.datetime(2010, 12, 1)
end = datetime.datetime(2010, 12, 30, 23, 59, 59)
step = datetime.timedelta(days=3)
for timestamp,used_percent in zip(data['t'],data['d']):
    #将数字时间转换为能够识别的时间 

    time_series.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
    mydata.append(used_percent)
    dt+=step

mydata=[float(x) for x in mydata]

dta=pd.Series(mydata)
dta.index = pd.Index(time_series)
dta.index = pd.DatetimeIndex(dta.index)
#dta = dta.resample('H')
#dta = dta.fillna(np.mean(mydata))

dta.plot()
plt.show()
#print(dta)
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)
#一阶差分
diff1 = dta.diff(1)
diff1.plot(ax=ax1)
plt.show()
#平稳性检测函数ADF
#from statsmodels.tsa.stattools import adfuller as ADF
#diff=0
#adf=ADF(dta)
#while adf[1]>=0.05:
#    diff=diff+1
#    adf=ADF(dta)
#print(u'原始序列经过%s阶差分后归于平稳,p值为%s' %(diff, adf[1]))

model = ARIMA(mydata, order=(0,1,0)).fit(disp=0)
#output = model.forecast()
#print(output)

dta.resample('H')
dta = dta.fillna(np.mean(mydata))
print(dta)
model = ARIMA(dta, (0,1,0), freq='H').fit(trend='nc')

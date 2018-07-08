#!/bin/python
#this script help you to analyse the input data(as pandas series), output the suitable parameter for ARIMA model

import pandas as pd
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import numpy
import sys
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

# 差分操作,d代表差分序列，比如[1,1,1]可以代表3阶差分。  [12,1]可以代表第一次差分偏移量是12，第二次差分偏移量是1
def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list #这个序列在恢复过程中需要用到
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        print(last_data_shift_list)
        shift_ts = tmp_ts.shift(i)
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts

def adf_test(timeseries):
    #print( 'Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

def analyst(ts):
    #first output original statictis:
    adf_test(ts)
    ts_log=numpy.log(ts)

def check_stable(ts):
    flag=True
    if(test_r['p-value']>0.05):
        print('P value large than 0.05: ',test_r['p-value'])
        flag=False
    if(test_r['Test Statistic']>test_r['Critical Value (5%)']):
        print('Test Statistic large than ',test_r['Critical Value (5%)'],' : ',test_r['Test Statistic'])
        flag=False
    if(not flag):
        print('failed stable checking!')
    return flag

if __name__=='__main__':
    print('start testing for the analyst program:')
    dateparse = lambda dates:datetime.strptime(dates,'%Y-%m')
    data=read_csv('../airpassengers/AirPassengers.csv',parse_dates=[0],index_col=0,date_parser=dateparse);
    ts=data['#Passengers']
    with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        print(ts)
    print(type(ts))
    #pyplot.plot(data)
    #pyplot.show()
    test_r=adf_test(ts)
    print(test_r['Test Statistic'])
    print(test_r['p-value'])
    print(test_r['#Lags Used'])
    print(test_r['Number of Observations Used'])
    print(test_r['Critical Value (1%)'])
    print(test_r['Critical Value (5%)'])
    print(test_r['Critical Value (10%)'])
    if(check_stable(test_r)):
        print('original series is stable.')
        exit 

    ts_log=numpy.log(ts)

    ts_log_diff=diff_ts(ts_log, [1])
    test_r=adf_test(ts_log_diff)
    print(test_r)
    if(check_stable(test_r)):
        print('series is stable after 1 log, 1 diff.')
        exit 

    ts_log_diff=diff_ts(ts_log, [1,1])
    test_r=adf_test(ts_log_diff)
    print(test_r)
    if(check_stable(test_r)):
        print('series is stable after 1 log, 2 diff.')
        exit 

    ts_log_diff=diff_ts(ts_log, [1,1,1])
    test_r=adf_test(ts_log_diff)
    print(test_r)
    if(check_stable(test_r)):
        print('series is stable after 1 log, 3 diff.')
        exit 
    
    print('series failed to be stable after 1 log, 3 diff.')

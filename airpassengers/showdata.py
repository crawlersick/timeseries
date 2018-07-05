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

# 还原操作
def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, numpy.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i-1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data # return numpy.exp(tmp_data)也可以return到最原始，tmp_data是对原始数据取对数的结果




dateparse = lambda dates:datetime.strptime(dates,'%Y-%m')
data=read_csv('AirPassengers.csv',parse_dates=[0],index_col=0,date_parser=dateparse);
#data=read_csv('AirPassengers.csv');
#print(data.head())
ts=data['#Passengers']
#pyplot.plot(data)
#pyplot.show()


def rolling_statistics(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).mean()

    #Plot rolling statistics:
    orig = pyplot.plot(timeseries, color='blue',label='Original')
    mean = pyplot.plot(rolmean, color='red', label='Rolling Mean')
    std = pyplot.plot(rolstd, color='black', label = 'Rolling Std')
    pyplot.legend(loc='best')
    pyplot.title('Rolling Mean & Standard Deviation')
    #pyplot.show(block=False)
    pyplot.show()

#rolling_statistics(data)

def adf_test(timeseries):
    #rolling_statistics(timeseries)#绘图
    print( 'Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

adf_test(ts)

ts_log=numpy.log(ts)
adf_test(ts_log)

#ts_log_diff=ts_log-ts_log.shift(periods=1)
#ts_log_diff.dropna(inplace=True)

ts_log_diff=diff_ts(ts_log,[1])

adf_test(ts_log_diff)

ts_log_diff2=ts_log_diff-ts_log_diff.shift(periods=1)
ts_log_diff2.dropna(inplace=True)
adf_test(ts_log_diff2)

pyplot.plot(ts_log_diff2)
pyplot.show()



def _proper_model(ts_log_diff, maxLag):
    best_p = 0
    best_q = 0
    best_model=None
    best_bic = 0
    for p in numpy.arange(maxLag):
        for q in numpy.arange(maxLag):
            model = ARMA(ts_log_diff, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARMA.bic
            print (bic, best_bic)
            if bic < best_bic:
                best_p = p
                best_q = q
                best_bic = bic
                best_model = results_ARMA
    return best_p,best_q,best_model
#p,q,bm=_proper_model(ts_log_diff, 10) #对一阶差分求最优p和q
#print(p)
#print(q)
# p=8, q=9

model=ARIMA(ts_log,order=(8,1,9))
results_ARIMA=model.fit(disp=-1)

pyplot.plot(ts_log_diff)
pyplot.plot(results_ARIMA.fittedvalues, color='red')#和下面这句结果一样
pyplot.plot(results_ARIMA.predict(), color='black')#predict得到的就是fittedvalues，只是差分的结果而已。还需要继续回退
pyplot.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
pyplot.show()

predict_ts=results_ARIMA.predict()
diff_recover_ts=predict_diff_recover(predict_ts,[1])
log_recover=numpy.exp(diff_recover_ts)
#绘图
#ts = ts[log_recover.index]#排除空的数据
pyplot.plot(ts,color="blue",label='Original')
pyplot.plot(log_recover,color='red',label='Predicted')
pyplot.legend(loc='best')
pyplot.title('RMSE: %.4f'% numpy.sqrt(sum((log_recover-ts)**2)/len(ts)))#RMSE,残差平方和开根号，即标准差
pyplot.show()


#forecast
# forecast方法会自动进行差分还原，当然仅限于支持的1阶和2阶差分
forecast_n=12
forecast_ARIMA_log=results_ARIMA.forecast(forecast_n)
forecast_ARIMA_log=forecast_ARIMA_log[0]
print(forecast_ARIMA_log)


#定义获取连续时间，start是起始时间，limit是连续的天数,level可以是day,month,year
import arrow
def get_date_range(start, limit, level='month',format='YYYY-MM-DD'):
    start = arrow.get(start, format)
    result=(list(map(lambda dt: dt.format(format) , arrow.Arrow.range(level,start,limit=limit))))
    dateparse2 = lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')
    return map(dateparse2, result)



# 预测从1961-01-01开始，也就是我们训练数据最后一个数据的后一个日期
new_index = get_date_range('1961-01-01', forecast_n)
forecast_ARIMA_log = pd.Series(forecast_ARIMA_log, copy=True, index=new_index)
print(forecast_ARIMA_log.head())

# 直接取指数，即可恢复至原数据
forecast_ARIMA = numpy.exp(forecast_ARIMA_log)
print(forecast_ARIMA)
pyplot.plot(ts,label='Original',color='blue')
pyplot.plot(forecast_ARIMA, label='Forcast',color='red')
pyplot.legend(loc='best')
pyplot.title('forecast')
pyplot.show()

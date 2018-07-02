from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import numpy
def autocorrelation(x,lags):#计算lags阶以内的自相关系数，返回lags个值，将序列均值、标准差视为不变
	n = len(x)
	x = numpy.array(x)
	variance = x.var()
	x = x-x.mean()
	result = numpy.correlate(x, x, mode = 'full')[-n+1:-n+lags+1]/\
		(variance*(numpy.arange(n-1,n-1-lags,-1)))
	return result

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series.tolist())
aa=autocorrelation(series.tolist(),5)
print(aa)
aa=autocorrelation(series.tolist(),10)
print(aa)
aa=autocorrelation(series.tolist(),15)
print(aa)

autocorrelation_plot(series)
pyplot.show()

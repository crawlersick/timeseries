import random
import pandas
from matplotlib import pyplot as plt
def type0():
    tempv=0
    y=[]
    y.append(tempv)
    for x in range(1,100):
        if(x%15==0):
            tempv=tempv/3*2
        else:
            d=random.randint(0,5)
            tempv=tempv+d
            y.append(tempv)
    st=pandas.Series(data=y)
    return st

if __name__=="__main__":
    st=type0()
    plt.plot(st)
    plt.show()

import random
import pandas
from matplotlib import pyplot as plt
def type1():
    tempv=0
    y=[]
    y.append(tempv)
    for x in range(100):
        if(x==20):
            tempv=tempv/2
        elif(x==80):
            tempv=tempv/7*4
        else:
            d=random.randint(0,5)
            tempv=tempv+d
            y.append(tempv)
    st=pandas.Series(data=y)
    return st

if __name__=="__main__":
    st=type1()
    plt.plot(st)
    plt.show()

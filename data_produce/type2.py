import random
import pandas
from matplotlib import pyplot as plt
def type2():
    tempv=0
    y=[]
    for x in range(100):
        if(x%10==0):
            d=random.randint(0,5)
            if(d==5 or d==4):d=0
            tempv=tempv+d
        y.append(tempv)
    st=pandas.Series(data=y)
    return st

if __name__=="__main__":
    st=type2()
    plt.plot(st)
    plt.show()

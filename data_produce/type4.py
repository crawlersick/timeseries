import random
import pandas
from matplotlib import pyplot as plt
def type4():
    tempv=0
    y=[]
    flag=1
    y.append(tempv)
    for x in range(1,100):
        if(x%5==0):
            flag=flag*-1
            tempv=tempv+random.randint(3,9)
        d=random.randint(3,9)
        tempv=tempv+(d*flag)
        y.append(tempv)
    st=pandas.Series(data=y)
    return st

if __name__=="__main__":
    st=type4()
    plt.plot(st)
    plt.show()

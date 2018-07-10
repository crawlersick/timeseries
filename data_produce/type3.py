import random
import pandas
from matplotlib import pyplot as plt
def type3():
    tempv=0
    y=[]
    d=random.randint(20,80)
    for x in range(100):
        y.append(d)
    st=pandas.Series(data=y)
    return st

if __name__=="__main__":
    st=type3()
    plt.plot(st)
    plt.show()

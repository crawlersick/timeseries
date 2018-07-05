import pandas as pd
df=pd.DataFrame([1,2,3,4,5],
        columns=['cols'],
        index=['a','b','c','d','e'])
print(df)

df2=pd.DataFrame([[1,2,3],[4,5,6]],
        columns=['col1','col2','col3'],
        index=['a','b'])

print(df2)
print(df2.index)
print(df2.values)
print(df.values)

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/PCAs.csv')
print df.shape

for i,col in enumerate(df.columns):
	print i,col

X = df.iloc[:,1:56]
y = df.iloc[:,57].values.reshape(-1,1)

print X.shape
print y.shape

trX,teX,trY,teY = train_test_split(
	preprocessing.minmax_scale(X),
	preprocessing.minmax_scale(y),
	test_size = 0.2
)

print trX.shape
print teX.shape
print trY.shape
print teY.shape

np.save('data/npy/trainX',trX)
np.save('data/npy/trainY',trY)
np.save('data/npy/testX',teX)
np.save('data/npy/testY',teY)

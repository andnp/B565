import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

train1 = pd.read_csv('../input/small_train.csv')
train2 = pd.read_csv('../input/small_train2.csv')

train = pd.concat([train1, train2])
del train1, train2

# np.random.shuffle(train)
train = train.reset_index()

Xtrain = train.ix[0:120000, 2:]
Xtest = train.ix[120000:200000, 2:]
Ytrain = train.ix[0:120000, 1]
Ytest = train.ix[120000:200000, 1]

clf = MLPClassifier(hidden_layer_sizes=(400))
clf.fit(Xtrain, Ytrain)
p = clf.predict(Xtest)
tmp = np.subtract(Ytest.T, p)
tmp = tmp.T
print 1.0 - float(tmp.sum())/float(tmp.shape[0])

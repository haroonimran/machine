import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

#Getting data from the quandl website:
df = quandl.get('WIKI/GOOGL', api_key = 'wH-v7aNLUnnyJkbxZy48')
#print(df.head())

#Adjusted Prices reprsent incorporation of stock splits etc. so always use adjusted.
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']*100
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Close']*100
df = df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']] 


forecast_col = 'Adj. Close'
df.fillna(-99999,inplace = True)

#Setting the number of days of data that will be used to predict our forecast. The following will forecast based on most recent 10% of the data frame.
#Length of the data frame here = number of days of trade data available. Since each row = 1 day of ticker price.
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())

X = np.array(df.drop(['label'],1)) # dropping label column and saving to variable 'X'
y = np.array(df['label'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.2)

#Switch between Linear Regression SVM etc. Pick one and assign to clf.
clf = LinearRegression()
#clf = svm.SVR()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy)
print(forecast_out)
print (len(X))
print("lenth of xtrain:",len(X_train))
print("lenth of xtest:",len(X_test))




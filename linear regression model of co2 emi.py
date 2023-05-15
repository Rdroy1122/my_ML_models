import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import math
import statistics


df=pd.read_csv("FuelConsumptionCo2.csv")
#dta frame
dff=df[["ENGINESIZE","CO2EMISSIONS"]]

# training of our linear_model

train_x=np.asanyarray(dff[["ENGINESIZE"]])
train_y=np.asanyarray(dff[["CO2EMISSIONS"]])

regr=linear_model.LinearRegression()
regr.fit(train_x,train_y)

print("coefficient:",regr.coef_)
print("intercept:",regr.intercept_)

#testing of the linear_model
test_x=np.asanyarray(dff[["ENGINESIZE"]].head(5))
test_y=np.asanyarray(dff[["CO2EMISSIONS"]].head(5))

test_y_=regr.predict(test_x)

rmse=math.sqrt(statistics.mean((test_y.flatten()-test_y_.flatten())**2))  # flattern is usded because we can't convert type 'ndarray' to numerator/denominator
test_y_=test_y_-rmse

print("RMSE:",rmse)
print("R2-score: %.2f" % r2_score(test_y , test_y_) )
print("actual:" ,test_y)
print("predicted:",test_y_)


plt.scatter(dff.ENGINESIZE, dff.CO2EMISSIONS, color='red')
plt.plot(train_x,regr.coef_[0][0]*train_x + regr.intercept_[0],'b')
plt.show()

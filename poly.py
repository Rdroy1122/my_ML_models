from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df=pd.read_csv(input("Enter file Name"))
cdf=df[['ENGINESIZE','CO2EMISSIONS']]

x=np.asanyarray(cdf[['ENGINESIZE']])
y=np.asanyarray(cdf[['CO2EMISSIONS']])

poly=PolynomialFeatures(degree=2)
x_=poly.fit_transform(x)

regr=linear_model.LinearRegression()
regr.fit(x_,y)

y__=regr.predict(x_) # perdiction of testing sample for r2 r2_score


print("coefficient:",regr.coef_)
print("intercept:",regr.intercept_)
print("R2-score: %.2f" % r2_score(y,y__))

y__=regr.predict(poly.fit_transform(np.asanyarray([[int(input("Enter engine size:"))]])))
print("predcited:",y__)

plt.scatter(cdf[['ENGINESIZE']],cdf[['CO2EMISSIONS']], color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = regr.intercept_[0]+ regr.coef_[0][1]*XX+ regr.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

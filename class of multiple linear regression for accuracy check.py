import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import math
import statistics

class Mylinear_model:

   def __init__ (self,a,b,c):
     self.x=a
     self.y=b
     self.z=c

   def predict_value(self):

    file=input("Enter File name:")
    df=pd.read_csv(file)
        #dta fram

    dff=df[[self.x,self.y,self.z]]

    train_x=np.asanyarray(dff[[self.x,self.z]])
    train_y=np.asanyarray(dff[[self.y]])

    regr=linear_model.LinearRegression()
    regr.fit(train_x,train_y)

    print("coefficient:",regr.coef_)
    print("intercept:",regr.intercept_)


      #testing of the linear_model
    test_x=np.asanyarray(dff[[self.x,self.z]].head(5))
    test_y=np.asanyarray(dff[[self.y]].head(5))

    test_y_=regr.predict(test_x)

    rmse=math.sqrt(statistics.mean((test_y.flatten()-test_y_.flatten())**2))  # flattern is usded because we can't convert type 'ndarray' to numerator/denominator
    test_y_=test_y_-rmse

    print("RMSE:",rmse)
    print("R2-score: %.2f" % r2_score(test_y , test_y_) )
    print("actual Value:" ,test_y)
    print("predicted Value:",test_y_)


    plt.scatter(dff[[self.x]], dff[[self.y]], color='red')
    plt.scatter(dff[[self.z]], dff[[self.y]], color='red')
    plt.plot(train_x,regr.coef_[0][0]*train_x + regr.intercept_[0],'b')
    plt.show()

c1=input("Enter Independent Variable :")
c3=input("Enter Independent Variable :")
c2=input("Enter Dependent Variable(Predict Value):")


co2emission=Mylinear_model(c1,c2,c3)
co2emission.predict_value()

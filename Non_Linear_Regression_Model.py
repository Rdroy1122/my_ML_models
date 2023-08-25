import numpy as np
import matplotlib.pyplot as plt

#Although linear regression can do a great job at modeling some datasets,
#it cannot be used for all datasets. First recall how linear regression, models a dataset.
# It models the linear relationship between a dependent variable y and the independent variables x. It has a simple equation, of degree 1, for example y =2x+ 3.

x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
x = np.arange(-5.0, 5.0, 0.1)
#Non-linear regression is a method to model the non-linear relationship between the independent variables  xand the dependent variable  y. Essentially any relationship that is not linear can be termed as non-linear, and
#is usually represented by the polynomial of  k degrees (maximum power of  x ). For example:y=ax**3+bx**2+cx+d

#Non-linear functions can have elements like exponentials, logarithms, fractions, and so on. For example:y=log(x)

#We can have a function that's even more complicated such as :y=log(ax**3+bx**2+cx+d)

#Let's take a look at a cubic function's graph.

x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

#Let's take a look at a quadratic function's graph.  Y=X**2    

x = np.arange(-5.0, 5.0, 0.1)


y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

#Exponential    Y=a+bc**X   
 #where b ≠0, c > 0 , c ≠1, and x is any real number. The base, c, is constant and the exponent, x, is a variable.

X = np.arange(-5.0, 5.0, 0.1)



Y= np.exp(X)

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


#Logarithmic  y=log(x)  

X = np.arange(-5.0, 5.0, 0.1)

Y = np.log(X)

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


#Sigmoidal/Logistic Y=a+b/1+c**(X−d)

X = np.arange(-5.0, 5.0, 0.1)


Y = 1-4/(1+np.power(3, X-2))

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

### Choosing a model

#From an initial look at the plot, we determine that the logistic function could be a good approximation,
#since it has the property of starting with a slow growth, increasing growth in the middle, and then decreasing again at the end; as illustrated below:

import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np

def readData (filename):
    df= pd.read_csv(filename)
    return df.values #returns the values as array elements
    
def hypothesis (theta, x):
    return theta[0] + (theta[1]*x)

def error(X,Y, theta):
    total_error = 0
    m= X.shape[0]
    for i in range (m):
        total_error += ((Y[i] - hypothesis (theta,X[i]))**2)

    return total_error

def gradient (X, Y, theta):
    grad = np.array([0.0,0.0])
    m= X.shape[0]
    
    for i in range (m):
        grad[0] += (hypothesis(theta, X[i])- Y[i])
        grad[1] += (hypothesis(theta, X[i]) - Y[i])*X[i]
    return grad
    
def gradientDescent(X,Y,learning_rate,maxItr):
    #grad = np.array([0.0,0.0])
    theta = np.array([0.0,0.0])
    e=[]
    for i in range (maxItr):
        grad = gradient(X,Y,theta)
        ce=error(X,Y,theta)
        e.append (ce)
        theta[0]= theta[0] - (learning_rate * grad[0])
        theta[1] = theta[1] - (learning_rate * grad[1])
    return theta,e


#training data
x= readData ('.../datasets/linear regression/linearX.csv')
x=x.reshape ((99,))
y =readData('.../datasets/linear regression/linearY.csv')
y=y.reshape((99,))

#data normalization
x= x - (x.mean()/(x.std()))
X=x
Y=y
plt.scatter(X,Y)

theta, e = gradientDescent(X,Y,learning_rate= 0.001, maxItr= 370)

def get_hypo_error(X):
    hypo_array=[]
    m= X.shape[0]
    for i in range (m):
        hypo_array.append(hypothesis(theta, X[i]))
    return hypo_array

hypothesis_array = get_hypo_error(X)
plt.scatter(X, hypothesis_array, color='orange')
plt.plot(X, hypothesis_array, color='r')
plt.show()

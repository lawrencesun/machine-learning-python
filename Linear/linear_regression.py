#coding=utf-8
# machine learning practice
# Yuliang Sun, 2015
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
import re
####################
## Feature Scaling
def feature_scaling(f):
	f_mean = np.mean(f)
	f_std = np.std(f)
	f_scaling = (f - f_mean)/f_std
	return (f_mean, f_std, f_scaling)
##
####################

######################
## Cost Function
def cost(X, y, theta):
	h = X * theta
	error = (h - y)
	J = error.T * error / (2*m)
	return J
##
######################

######################
## Normal Equations, no need to feature scaling
def normal_equations(x, y): 
	theta = (x.T * x)**(-1) * x.T * y
	return theta
##
######################

#####################
## Graditent Descent
def gradient_descent(alpha, theta, iters, X, y):
	i = 0
	J_list = []
	while i < iters:
		theta = theta - alpha / m * X.T * (X * theta - y)
		i += 1
		J = cost(X, y, theta)
		J_list.append(J[0,0])	
	return (theta, J_list)
##
#####################


####################
## Load Data
## male: gender = 1; female: gender = 2; 

datafile = pd.read_excel("psydata.xlsx")

features = datafile.loc[datafile['gender'] == 1].loc[0:500,['height','weight','vcapacity','50m','jump']]
target = datafile.loc[datafile['gender'] == 1].loc[0:500,['longdistance-m']]
#target = datafile.loc[datafile['gender'] == 2].loc[0:500,['longdistance-f']]

# number of features
n = 5
y = np.mat(target.values)
# number of training
m = y.size

####################
## get theta through normal equations

# add ones to X
X = np.c_[np.ones(m),np.mat(features)] 
theta_ne = normal_equations(X, y)
print 'theta through normal equations: \n%r\n' %theta_ne
J_ne = cost(X, y, theta_ne)
print 'Cost: %r\n' %J_ne[0,0]


#####################
## get theta through gradient descent

# feature scaling
(x_mean, x_std, x_scaling) = feature_scaling(features)
x_scaling = np.mat(x_scaling)
X = np.c_[np.ones(m),x_scaling]

# set alpha and iteration numbers
alpha = 0.01
theta_gd = np.zeros([(n + 1), 1])
iters = 1000

(theta_gd, J_gd) = gradient_descent(alpha, theta_gd, iters, X, y)
print 'theta through gradient descent: \n%r\n' %theta_gd
print 'Cost: %r\n' %J_gd[-1]

#####################
## Learning Curve

#for i in J_gd:  
#	print i
xData = np.arange(0,iters,1)
yData = J_gd
plt.title('Learning Curve', size=14)
plt.xlabel('iterations', size=14)
plt.ylabel('cost', size=14)
plt.plot(xData, yData, color='r', linestyle='-')
plt.show()


#####################
## Test
x_test = datafile.loc[datafile['gender'] == 1].loc[1000:1500,['height','weight','vcapacity','50m','jump']]
print 'Test Features: \n%r\n' %x_test

# number of test 
y_test = datafile.loc[datafile['gender'] == 1].loc[1000:1500,['longdistance-m']]
y_test = np.mat(y_test)
#m = x_test.shape[0]
m = y_test.size
X_test = np.c_[np.ones(m),x_test]

######################
## Prediction using theta_ne

y_new = X_test * theta_ne
print 'Prediction using theta_ne: \n%r\n' %y_new
print 'Error: \n%r\n' %(y_new - y_test)


#####################
## Prediction using theta_gd

# feature scaling
x_test = np.mat((x_test - x_mean) / x_std)
X_test = np.c_[np.ones(m),x_test]
y_new = X_test * theta_gd
print 'Prediction using theta_gd: \n%r\n' %y_new
print 'Error: \n%r\n' %(y_new - y_test)




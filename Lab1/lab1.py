import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from lab1Utils import *
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # fxn()


# Seed for reproducability
np.random.seed(0)
# import pylab
###########################################################################################
########################################Part 1#############################################
###########################################################################################

############################################################################
########            Data Generation: example                        ########
############################################################################
#rand_gauss
n=100
mu=[1,1]
sigma=[1,1]
rand_gauss(n,mu,sigma)
#rand_bi_gauss
n1=20
n2=20
mu1=[1,1]
mu2=[-1,-1]
sigma1=[0.9,0.9]
sigma2=[0.9,0.9]
data1=rand_bi_gauss(n1,n2,mu1,mu2,sigma1,sigma2)
#rand_clown
std1=1
std2=5
n1=50
n2=50
data2=rand_clown(n1,n2,std1,std2)
#rand_checkers
std=0.1
data3=rand_checkers(n1,n2,std)
# Data X would be data1 from now on
############################################################################
########            Displaying labeled data                         ########
############################################################################
plt.close("all")

plt.figure(1, figsize=(15,5))
plt.subplot(131)
plt.title('Rand Bi Gauss')
plot_2d(data1[:,:2],data1[:,2],w=None)

plt.subplot(132)
plt.title('Rand Clown')
plot_2d(data2[:,:2],data2[:,2],w=None)

plt.subplot(133)
plt.title('Rand Checkers')
plot_2d(data3[:,:2],data3[:,2],w=None)
# plt.show()
plt.savefig('q1a.png')
# pylab.savefig('q1.png')

###########################################################################################
########################################Part 2#############################################
###########################################################################################

############################################################################
########            Logistic regression                             ########
############################################################################
dataX=data1[:,:2]
dataY=data1[:,2]
from sklearn import linear_model
lr = linear_model.LogisticRegression()
lr.fit(dataX,dataY)
### Q3 Coeff and Intercept ####
print "Coeff: " + str(lr.coef_)
print "Intercept: " + str(lr.intercept_)
### Q4 Predict and Score ####
n1=10
n2=10
dataTest=rand_bi_gauss(n1,n2,mu1,mu2,sigma1,sigma2)
testX=dataTest[:,:2]
testY=dataTest[:,2]
predictClass = lr.predict(testX)
print "Score: "+ str(lr.score(testX,testY))
### Q5 Plotting ####
# grid design with meshgrid
xx,yy=grid_2d(dataX,50)
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Display the result into a color plot, first solution
plt.figure(2, figsize=(4,3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
labs=np.unique(dataY)
idxbyclass=[ np.where(dataY==labs[i])[0] for i in xrange(len(labs))]
for i in xrange(len(labs)):
    plt.scatter(dataX[idxbyclass[i],0],dataX[idxbyclass[i],1],
            color=collist[i%len(collist)],cmap=plt.cm.Paired,
            marker=symlist[i%len(symlist)],s=90,edgecolors='k')
# plt.show()
plt.savefig('q2a.png')
# Display the result into a color plot, second solution
plt.figure(3, figsize=(4,3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(dataX[:, 0], dataX[:, 1], c=dataY, edgecolors='k',
                cmap=plt.cm.Paired, s=100)
# plt.show()
plt.savefig('q2b.png')
# Display the result into a color plot, third solution
plt.figure(4)
plot_2d(dataX,dataY)
frontiere(lr.predict,dataX,step=100)
# plt.show()
plt.savefig('q2c.png')

### Q6 Real life dataset ####
import pandas as pd
dX = pd.read_csv('zip.train',delimiter=" ",header=None)
data4 = np.array(dX)
trainX = data4[:,1:257]
trainY = data4[:,0]
test = pd.read_csv('zip.test',delimiter=" ",header=None)
test = np.array(test)
testX = test[:,1:257]
testY = test[:,0]
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
OVR = OneVsRestClassifier(linear_model.LogisticRegression()).fit(trainX,trainY)
OVO = OneVsOneClassifier(linear_model.LogisticRegression()).fit(trainX,trainY)
print 'One vs rest accuracy: %.3f' % OVR.score(testX,testY)
print 'One vs one accuracy: %.3f' % OVO.score(testX,testY)


###########################################################################################
########################################Part 3#############################################
###########################################################################################


############################################################################
########                Perceptron example                          ########
############################################################################
from sklearn import linear_model

# MSE Loss
epsilon=0.01
niter=75

dataX=data1[:,:2]
dataY=data1[:,2]

## If using SGD Classifier
clf = linear_model.SGDClassifier()
clf.fit(dataX,dataY)

## If using SGD Algorithm
#w_ini: intial guess for the hyperplane
w_ini=np.zeros([niter,dataX.shape[1]+1])
std_ini=1
for i in range(dataX.shape[1]+1):
	w_ini[-1,-i+1]=std_ini*np.random.randn(1,1)
	print w_ini[-1,-i+1]

lfun=mse_loss
gr_lfun=gr_mse_loss

plt.figure(7)
wh,costh=gradient(dataX,dataY,epsilon,niter,w_ini,lfun,gr_lfun,stoch=False)
plot_gradient(dataX,dataY,wh,costh,lfun)
plt.suptitle('MSE and batch')
# plt.show()
plt.savefig('msebatch.png')


epsilon=0.001
plt.figure(8)
plt.suptitle('MSE and stochastic')
wh_sto,costh_sto=gradient(dataX,dataY,epsilon,niter,w_ini,lfun,gr_lfun,
						stoch=True)
plot_gradient(dataX,dataY,wh_sto,costh_sto,lfun)
# plt.show()
plt.savefig('msestoch.png')

# Hinge Loss
epsilon=0.01
niter=30

dataX=data1[:,:2]
dataY=data1[:,2]

w_ini=np.zeros([niter,dataX.shape[1]+1])
std_ini=10
for i in range(dataX.shape[1]+1):
	w_ini[-1,-i+1]=std_ini*np.random.randn(1,1)


lfun=hinge_loss
gr_lfun=gr_hinge_loss
wh,costh=gradient(dataX,dataY,epsilon,niter,w_ini,lfun,gr_lfun,stoch=False)

plt.figure(9)
plt.suptitle('Hinge and batch')
plot_gradient(dataX,dataY,wh,costh,lfun)
# plt.show()
plt.savefig('hingebatch.png')

plt.figure(10)
plt.suptitle('Hinge and stochastic')
wh_sto,costh_sto=gradient(dataX,dataY,epsilon,niter,w_ini,lfun,gr_lfun,
						stoch=True)
plot_gradient(dataX,dataY,wh_sto,costh_sto,lfun)
# plt.show()
plt.savefig('hingestoch.png')



# Create a figure with all the boundary displayed with a
# brighter display for the newest one
epsilon=1
niter=30
plt.figure(11)
wh_sto,costh_sto=gradient(dataX,dataY,epsilon,niter,w_ini,lfun,gr_lfun,
						stoch=True)
indexess=np.linspace(0,1,niter)
for i in range(niter):
	plot_2d(dataX,dataY,wh_sto[i,:],indexess[i])
plt.savefig('plot2d.png')

############################################################################
########                Perceptron for larger dimensions            ########
############################################################################

epsilon=0.01
niter=50

dataX=data2[:,:2]
dataY=data2[:,2]

proj=poly2;
dataXX=proj(dataX)
w_ini=np.random.randn(niter,dataXX.shape[1]+1)

clf = linear_model.SGDClassifier()
clf.fit(dataXX, dataY)

plt.ion()
plt.figure(11)
plt.clf()
plot_2d(dataX,dataY)
frontiere(lambda xx:clf.predict(poly2(xx)),dataX)
plt.draw()
plt.show()
plt.savefig('q3part4.png')



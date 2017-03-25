"""
@author: Shubham
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
# from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# discriminant_analysis.LinearDiscriminantAnalysis

def rand_gauss(n=100,mu=[1,1],sigma=[0.1,0.1]):
    """ Sample n points from a Gaussian variable with center mu, 
    and std deviation sigma
    """
    d=len(mu)
    res=np.random.randn(n,d)
    return np.array(res*sigma+mu)


def rand_bi_gauss(n1=100,n2=100,mu1=[1,1],mu2=[-1,-1],sigma1=[0.1,0.1],
                sigma2=[0.1,0.1]):
    """ Sample n1 and n2 points from two Gaussian variables centered in mu1,
    mu2, with std deviation sigma1, sigma2
    """
    ex1=rand_gauss(n1,mu1,sigma1)
    ex2=rand_gauss(n2,mu2,sigma2)
    res=np.vstack([np.hstack([ex1,np.transpose(np.ones([1,n1]))]),
    np.hstack([ex2,np.transpose(-1*np.ones([1,n2]))])])
    ind=range(res.shape[0])
    np.random.shuffle(ind)
    return np.array(res[ind,:])

def grid_2d(dataX,step=20):
	xmin,xmax=dataX[:,0].min() -1, dataX[:,0].max()+1
	ymin,ymax=dataX[:,1].min() -1, dataX[:,1].max()+1
	xx,yy=np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step),
						np.arange(ymin,ymax,(ymax-ymin)*1./step))
	
	return [xx,yy]

symlist=['o','s','+','x','D','*','p','v','-','^']
collist=['blue','red','purple','orange','salmon','black','grey','fuchsia']

n1=20
n2=20
mu1=[1,1]
mu2=[-1,-1]
sigma1=[0.9,0.9]
sigma2=[0.9,0.9]
data1=rand_bi_gauss(n1,n2,mu1,mu2,sigma1,sigma2)

dataX=data1[:,:2]
dataY=data1[:,2]

# lda = LDA()
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)


### For splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.5, random_state=42)

# lda.fit(X_train,y_train)
# y_pred = lda.predict(X_test)
# print "LDA results:"
# print metrics.classification_report(y_test, lda.predict(X_test))

lda.fit(dataX,dataY)
lda.coef_
lda.intercept_

print "LDA coef:" + str(lda.coef_)
print "LDA intercept:" + str(lda.intercept_)
print "LDA results:"
print metrics.classification_report(dataY, lda.predict(dataX))


#meshgrid creation for visualization
xx,yy=grid_2d(dataX,50)
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
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

# Display the result into a color plot, second solution
plt.figure(3, figsize=(4,3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(dataX[:, 0], dataX[:, 1], c=dataY, edgecolors='k',
				cmap=plt.cm.Paired, s=100)
plt.show()
"""
=========================================
SVM: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a Support Vector Machine classifier with
linear kernel.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
# we create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20
# fit the model
# Primal form 
for t,tol in enumerate((0.1, 0.001, 0.0001)):
	clf = svm.LinearSVC(dual=False,tol = tol)
	clf.fit(X, Y)
	# Dual form 
	clf1 = svm.LinearSVC(dual=True,tol = tol)
	clf1.fit(X, Y)
	# get the separating hyperplane
	w = clf.coef_[0]
	a = -w[0] / w[1]
	w_intercept = clf.intercept_
	w_all = np.append(w,w_intercept)

	w1 = clf1.coef_[0]
	a1 = -w1[0] / w1[1]
	w1_intercept = clf1.intercept_
	w1_all = np.append(w1,w1_intercept)

	from sklearn.metrics import euclidean_distances
	distances = euclidean_distances(w, w1)
	print "tolerance = " + str(tol)
	print "Euclidean distance between coeff: " + str(distances)
	print "Absolute distance between coeff: " + str(np.abs(w-w1))
	distances = euclidean_distances(w_all, w1_all)
	print "Euclidean distance between weights (coeff+intercept): " + str(distances)
	print "Absolute distance between coeff: " + str(np.abs(w_all-w1_all))


# xx = np.linspace(-5, 5)
# yy = a * xx - (clf.intercept_[0]) / w[1]

# # plot the parallels to the separating hyperplane that pass through the
# # support vectors
# b = clf.support_vectors_[0]
# yy_down = a * xx + (b[1] - a * b[0])
# b = clf.support_vectors_[-1]
# yy_up = a * xx + (b[1] - a * b[0])

# # plot the line, the points, and the nearest vectors to the plane
# plt.plot(xx, yy, 'k-')
# plt.plot(xx, yy_down, 'k--')
# plt.plot(xx, yy_up, 'k--')

# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
#             s=80, facecolors='none')
# plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

# plt.axis('tight')
# plt.show()

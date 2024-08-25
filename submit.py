import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

clf = LogisticRegression(C = 10, tol=0.1, solver='lbfgs', penalty='l2', max_iter=300)


################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
    X_train = my_map(X_train)
    clf.fit(X_train, y_train)
    w = clf.coef_[0]  # Access the coefficients via the named_steps attribute
    b = clf.intercept_[0]
    
    return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    d = X.shape[0]
    u = np.ones((d,1))
    X = np.concatenate((X, u), axis=1)
    n = X.shape[1]
    n = np.shape(X)[1]
    X = np.cumprod(np.flip(2 * X - 1, axis=1), axis=1)
    feat = khatri_rao(X.T, X.T).T
    upp_mat = np.triu(np.ones((n, n)), k=0)-np.eye(n)
    u1d = upp_mat.flatten().astype(bool)
    feat = feat[:,u1d]
    
    return feat

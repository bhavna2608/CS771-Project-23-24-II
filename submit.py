import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
    x_train=my_map(X_train)
    # model = LinearSVC(C=0.15,max_iter=105)
    model = LogisticRegression(C=10, solver='lbfgs',penalty='l2', max_iter=300, tol = 0.1, dual="auto")
    model.fit(x_train, y_train)
    w = model.coef_
    b = model.intercept_
    w=w.T[:,-1]

    return w, b

def my_map( X ):
  feat=[]
  for row in X:
    X = np.array([row])
    mapp_X = 1-2*X
    X = np.flip(np.cumprod( np.flip( mapp_X , axis = 1 ), axis = 1 ), axis=1)
    B = []
    for row in X:
      for i in range(len(row)):
        for j in range(i , len(row)):
          product = row[i] * row[j]
          if i == j:
            product = (row[i])
          B.append(product)
    feat.append([np.array(B)])

  feat=np.concatenate(feat, axis=0)
  return feat

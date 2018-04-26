# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# getting path of raw data and train/test data
processeddata_path = os.path.join(os.path.pardir,'data','processed')
X_train = pd.read_csv(os.path.join(processeddata_path,'X_train.csv')).values
y_train = pd.read_csv(os.path.join(processeddata_path,'y_train.csv')).values
X_test =  pd.read_csv(os.path.join(processeddata_path,'x_test.csv')).values
y_test =  pd.read_csv(os.path.join(processeddata_path,'y_test.csv')).values

X_train = X_train[:,1:]
X_test = X_test[:,1:]
# Fitting the Logistic Regression to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# predicting the Test set results
y_pred = classifier.predict(X_test)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# Visualizing the test Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set[:,0] == j, 0], X_set[y_set[:,0] == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
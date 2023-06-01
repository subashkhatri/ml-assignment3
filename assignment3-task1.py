# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load the data
train_df = pd.read_csv('Datasets/train.csv')
test_df = pd.read_csv('Datasets/test.csv')
sample_df = pd.read_csv('Datasets/sample_submission.csv')

# We are going to use TfidfVectorizer to convert the raw tweets into vectors of TF-IDF features.
vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])
y_train = train_df['target']

# Visualization of the distribution of the target variable in the training set
sns.countplot(x='target', data=train_df)
plt.title('Distribution of Disaster Tweets')
plt.show()

# Training SGDClassifier. We set the loss function to 'log' to apply logistic regression and the penalty to 'l2' for ridge regularization.
clf_sgd = SGDClassifier(loss='log', penalty='l2', random_state=42)
clf_sgd.fit(X_train, y_train)

# predict the labels on test dataset
y_pred_sgd = clf_sgd.predict(X_test)

# Calculating metrics for the SGDClassifier model
precision_sgd = precision_score(sample_df['target'], y_pred_sgd)
recall_sgd = recall_score(sample_df['target'], y_pred_sgd)
f1_sgd = f1_score(sample_df['target'], y_pred_sgd)

# Printing SGDClassifier scores
print('Precision for SGDClassifier: ', precision_sgd)
print('Recall for SGDClassifier: ', recall_sgd)
print('F1 Score for SGDClassifier: ', f1_sgd)

# Confusion matrix for SGD
cm_sgd = confusion_matrix(sample_df['target'], y_pred_sgd)
sns.heatmap(cm_sgd, annot=True, fmt='d')
plt.title('Confusion matrix for SGD')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Mini-batch gradient descent
from sklearn.linear_model import SGDClassifier
clf_mb = SGDClassifier(loss='log', random_state=42, max_iter=1000, tol=1e-3)
clf_mb.fit(X_train, y_train)
y_pred_mb = clf_mb.predict(X_test)

# Calculating metrics for the Mini-batch model
precision_mb = precision_score(sample_df['target'], y_pred_mb)
recall_mb = recall_score(sample_df['target'], y_pred_mb)
f1_mb = f1_score(sample_df['target'], y_pred_mb)

# Printing Mini-batch GD scores
print('Precision for Mini-batch GD: ', precision_mb)
print('Recall for Mini-batch GD: ', recall_mb)
print('F1 Score for Mini-batch GD: ', f1_mb)

# Confusion matrix for Mini-batch GD
cm_mb = confusion_matrix(sample_df['target'], y_pred_mb)
sns.heatmap(cm_mb, annot=True, fmt='d')
plt.title('Confusion matrix for Mini-batch GD')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

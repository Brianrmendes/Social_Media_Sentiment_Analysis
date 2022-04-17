import pandas as pd
import pickle as pickle

df = pd.read_csv('tests/new.csv')
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.5, random_state=24)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(ctmTr, y_train)
lr_score = lr.score(X_test_dtm, y_test)
y_pred_lr = lr.predict(X_test_dtm)
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()

#True positive and true negative rates
tpr_lr = round(tp/(tp + fn), 4)
tnr_lr = round(tn/(tn+fp), 4)
# df = pd.DataFrame(list(zip(tweet_list, q)),columns =['Tweets', 'sentiment'])
pickle.dump(lr,open('model.pkl','wb'))

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:59:39 2019

@author: dbda
"""


import pandas as pd

data=pd.read_json("file:///F:/PROJECT/Final Models/all_source_title_5080.json")

data=data.dropna()


cates = data.groupby('category')
print("Total categories:", cates.ngroups)
allsize = cates.size()
print(allsize)


from sklearn.preprocessing import LabelEncoder
lbcode = LabelEncoder()
trny = lbcode.fit_transform(y)
print(trny)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


X = data[['article', 'title','source']]
y = data['category']
from sklearn.preprocessing import LabelEncoder
lbcode = LabelEncoder()
y = lbcode.fit_transform(y)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, stratify=data[['category', 'source']], random_state=2019)



cates = y_train.groupby(y_train)
print("Total categories:", cates.ngroups)
allsize = cates.size()
print(allsize)



#########   X   ###########


tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=3000)


#created sparse matrix of tfidf values of aricle column
#and converted the matrix to dataframe
tfidf_vect.fit(X.iloc[:,0])
xtrain_tfidf_article =  pd.DataFrame(tfidf_vect.transform(X_train.iloc[:,0]).todense(),columns=tfidf_vect.get_feature_names())
xtest_tfidf_article =  pd.DataFrame(tfidf_vect.transform(X_test.iloc[:,0]).todense(),columns=tfidf_vect.get_feature_names())


#created sparse matrix of tfidf values of Title column
#and converted the matrix to dataframe
tfidf_vect.fit(X.iloc[:,1])
xtrain_tfidf_title =  pd.DataFrame(tfidf_vect.transform(X_train.iloc[:,1]).todense(),columns=tfidf_vect.get_feature_names())
xtest_tfidf_title =  pd.DataFrame(tfidf_vect.transform(X_test.iloc[:,1]).todense(),columns=tfidf_vect.get_feature_names())


#concatenated tile and article martices
xtrain_tfidf=pd.concat([xtrain_tfidf_article,xtrain_tfidf_title],axis=1)
xtest_tfidf=pd.concat([xtest_tfidf_article,xtest_tfidf_title],axis=1)
print(xtrain_tfidf.shape)
print(xtest_tfidf.shape)
print(y_train.shape)
print(y_test.shape)
##### o/p
#print(xtrain_tfidf.shape)
#(3556, 6000)
#
#print(xtest_tfidf.shape)
#(1524, 6000)
#
#print(y_train.shape)
#(3556,)
#
#print(y_test.shape)
#(1524,)

lreg = LogisticRegression(random_state=2019,C=100).fit(xtrain_tfidf, y_train)

# save/dump the model here, use it for testing on a separate data
#from joblib import dump, load
#dump(lreg, 'tfidf_logisticreg_combine.joblib') 


y_pred_lreg = lreg.predict(xtest_tfidf)
acc_lreg = accuracy_score(y_test, y_pred_lreg)
print("LR, WordLevel TF-IDF: ", acc_lreg)
print(confusion_matrix(y_test, y_pred_lreg))
print(classification_report(y_test, y_pred_lreg))



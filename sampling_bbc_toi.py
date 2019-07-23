# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:27:58 2019

@author: dbda
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#merge = pd.read_json("file:///D:/PROJECT/MERGED_JSONFILES.json")

toi_data=pd.read_json("file:///D:/PROJECT/Final Models/TOI_FINAL_3004news.json")
bbc_data = pd.read_json("file:///D:/PROJECT/Final Models/bbc_title_2168.json") 

cates = bbc_data.groupby('category')
allsize = cates.size()
#print("Total categories:", cates.ngroups)
#print(allsize)


# newdf : traindata, with around 360 rows of each category, from BBC data
newdf = pd.DataFrame()
for name, grp in cates:
    newdf = pd.concat([newdf,grp.iloc[0:360,:]])
    

  
toirem = pd.DataFrame()
#for name, grp in cattoi:
#    newdf = pd.concat([newdf,grp.iloc[0:100,:]])
#    toirem = pd.concat([toirem, grp.iloc[100:,:]])

cattoi = toi_data.groupby('category')
print("Total categories:", cattoi.ngroups)
alltoi = cattoi.size()
print(alltoi)
#### --- Random Sampling
#### --- TOI data mixing with BBC for training,around 20 rows of each category, 20*5 = 100
#### --- #  : testdata, all remaining rows after merging into the training data
for name, grp in cattoi:
    train = grp.sample(n=20,replace=False)
    newdf = pd.concat([newdf,train])
    indxtrain = grp.index.difference(train.index)
    toirem = pd.concat([toirem, grp.loc[indxtrain,:]])
    
    
#### --- Get only few samples for testing, eg, 50 rows from each category  
catrem = toirem.groupby('category')
alltoirem = catrem.size()
#print("Total categories:", catrem.ngroups)
#print(alltoirem)

testdf = pd.DataFrame()
for name, grp in catrem:
    train = grp.sample(n=50,replace=False)
    testdf = pd.concat([newdf,train])


# TRAIN Data, only one column
X = newdf['article']
y = newdf['category']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, stratify= y ,random_state=2019)

#     X
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, stop_words='english')
tfidf_vect.fit(X)
#print(tfidf_vect.vocabulary_)
xtrain_tfidf =  tfidf_vect.transform(X)

# encoding y
from sklearn.preprocessing import LabelEncoder
lbcode = LabelEncoder()
y = lbcode.fit_transform(y)


lreg = LogisticRegression(random_state=2019,C=100, multi_class="multinomial").fit(xtrain_tfidf, y)

### here we can save/dump our model and use for further testing on separate data

# TEST DATA
x_test = toirem['article']
y_test = toirem['category']

y_test = lbcode.fit_transform(y_test)
#tfidf_vect.fit(toi_data['article'])
xtest_tfidf =  tfidf_vect.transform(x_test)
#LR, WordLevel TF-IDF:0.9466571834992887

# fit-transform test data
#xtest_tfidf =  tfidf_vect.fit_transform(x_test)
# LR, WordLevel TF-IDF:  0.1906116642958748

y_pred_lreg = lreg.predict(xtest_tfidf)
acc_lreg = accuracy_score(y_test, y_pred_lreg)
print("LR, WordLevel TF-IDF: ", acc_lreg)
print(confusion_matrix(y_test, y_pred_lreg))
print(classification_report(y_test, y_pred_lreg))








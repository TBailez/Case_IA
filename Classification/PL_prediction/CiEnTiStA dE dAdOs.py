# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:47:06 2022

@author: tomas
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

dff = pd.read_csv('E0_2016.csv')
df = dff[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG',
       'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC',
       'AC', 'HY', 'AY', 'HR', 'AR']]


cols = ['HomeTeam', 'AwayTeam', 'FTR','HTR']

class MultiColumnLabelEncoder:
    

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode


    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self


    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)


    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output
multi = MultiColumnLabelEncoder(columns=cols)
   
df  = multi.fit_transform(df)

X = df.drop(['FTR'],1)
y = df['FTR']
cols.remove('FTR')

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify = y)

clf = xgb.XGBClassifier(seed=123,
                        eval_metric='mlogloss') 

parameters = { 'learning_rate' : [0.9],
               'n_estimators' : [30],
               'eta' : [0.001],
               'max_depth': [6],
               'min_child_weight': [3],
               'colsample_bytree' : [0.7],
                 }  
    
    
grid_obj = GridSearchCV(clf,
                        param_grid=parameters,
                        cv=10)
    
    
grid_obj = grid_obj.fit(X_train,y_train)   
clf = grid_obj.best_estimator_   
preds = clf.predict(X_test)
preds = pd.DataFrame(preds)
y_test = y_test.reset_index()
preds['y_test'] = y_test['FTR']
preds['ACC'] = preds[0] - preds['y_test']
vc_preds = preds['ACC'].value_counts()
total = vc_preds.sum()
total_acertos = vc_preds.loc[0]
total_erros = total - total_acertos
acc = total_acertos/total
print('Total de jogos:',total)
print('Total de acertos:',total_acertos)
print('Total de erros:',total_erros)
print('Precisão:',round(acc,4))
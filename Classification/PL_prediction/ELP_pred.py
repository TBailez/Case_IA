# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 09:50:24 2022

@author: tomas
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

#ALTERAR PARA E0_final

dff = pd.read_csv('E0_2016.csv')
df = pd.read_excel('final_dataset.xlsx')


df = df[['HomeTeam','AwayTeam','FTHG','FTAG','FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP',]]

renm = ['TimeCasa', 'TimeFora','GC', 'GF','RF', 'GMTC','GMTF', 'GTTC','GTTF', 'TCP','TFP']

df.columns = [renm]
dfs = df


cols = ['TimeCasa', 'TimeFora', 'RF']

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



X = df.drop(['GC','GF'],1)
y = df['GC']
a = df['GF']
cols.remove('RF')


#from sklearn.model_selection import train_test_split
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify = y)'''
def predict_labels(clf, features, target):
    
    
    y_pred = clf.predict(features)
        
    
    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))
    
def prediction(cf):
    X_train = X.head(310)
    X_test = X.tail(70)
    y_train = cf.head(310)
    y_test = cf.tail(70)

    '''
    
    housing_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    params = {"objective":"reg:squaredlogerror", "max_depth":3}
    
    eta_vals = [0.001, 0.01, 0.1]
    best_rmse = []
    
    for curr_val in eta_vals:
    
        params["eta"] = curr_val
        
    
        cv_results = xgb.cv(dtrain=housing_dmatrix,params=params,nfold=3,early_stopping_rounds = 5,num_boost_round = 10, metrics = "rmse",as_pandas=True, seed = 123)
    
        best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])
    
    print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))
    
    max_depths = [2,5,10,20]
    best_rmse = []
    
    
    for curr_val in max_depths:
    
        params["max_depth"] = curr_val
        
        cv_results = xgb.cv(dtrain=housing_dmatrix,params=params,nfold=2,early_stopping_rounds = 5,num_boost_round = 10, metrics = "rmse",as_pandas=True, seed = 123)
        
        best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])
    
    print(pd.DataFrame(list(zip(max_depths, best_rmse)),columns=["max_depth","best_rmse"]))
    
    '''

    parameters = { 'learning_rate' : [0.9],
                   'n_estimators' : [30],
                   'eta' : [0.001],
                   'max_depth': [6],
                   'min_child_weight': [3],
                   'colsample_bytree' : [0.7],
                 }  
    
    
    clf = xgb.XGBClassifier(seed=123,
                            eval_metric='mlogloss')
    
    
    f1_scorer = make_scorer(f1_score,pos_label='H')
    
    
    grid_obj = GridSearchCV(clf,
                            scoring=f1_scorer,
                            param_grid=parameters,
                            cv=10)
    
    
    grid_obj = grid_obj.fit(X_train,y_train)
    
    
    clf = grid_obj.best_estimator_
    
    preds = clf.predict(X_test)
    
    preds = preds.reshape(len(preds),1)
    
    accuracy = float(np.sum(preds==y_test))/y_test.shape[0]

    if cf.columns[0] == ('GC',):
        st = 'GC'
    elif cf.columns[0] == ('GF',):
        st = 'GF'
    X_test[st+'_previsão'] = preds
    X_test[st+'_real'] = y_test
    teste  = multi.inverse_transform(X_test)
    teste['res_'+st] = teste[(st+'_previsão',)]-teste[(    st+'_real',)]

    tt = teste[teste[(        'res_'+st,)]==0]
    return accuracy, tt, teste
#%%
acc_gc, gc_acertados, previsao_final_gc = prediction(y)
acc_gf, gf_acertados, previsao_final_gf = prediction(a)

previsao_final = previsao_final_gc.merge(previsao_final_gf, how='left')

rename = ['TimeCasa', 'TimeFora','RF', 'GMTC','GMTF', 'GTTC','GTTF', 'TCP','TFP',
        'GC_previsão','GC_real','res_GC', 'GF_previsão', 'GF_real','res_GF']

previsao_final.columns = [rename]
previsao_final = previsao_final[['TimeCasa', 'TimeFora','GMTC','GMTF',
                                 'GTTC','GTTF', 'TCP','TFP','GC_real',
                                 'GF_real', 'RF','GC_previsão', 'GF_previsão']]
                              
#%%

previsao_final['diff_gols'] = previsao_final[('GC_previsão',)] - previsao_final[ ('GF_previsão',)]

cond = [(previsao_final['diff_gols']>0),
        (previsao_final['diff_gols']<0),
        (previsao_final['diff_gols']==0)
        ]

choice = [2,0,1]

previsao_final['RF_previsão'] = np.select(cond,choice,'ERRO')
previsao_final[['RF_previsão']] = previsao_final[['RF_previsão']].apply(pd.to_numeric)

del previsao_final['diff_gols']

#%%
previsao_final['RF_RESULTADO'] = previsao_final[('RF_previsão',)]-previsao_final[(         'RF',)]
vc_final = previsao_final['RF_RESULTADO'].value_counts()
acertos = vc_final.iloc[0]
acc_final = acertos/70
print('Precisão de:',round(acc_final,5)*100,'%' )
#%%
'''
PLOT DE GRAFICOS COMPARANDO RESULTADOS
vc_real = previsao_final['RF'].value_counts()
vc_prev = previsao_final['RF_previsão'].value_counts()

def plotpie(df):

    df = pd.DataFrame(df,columns=['RESULTADO'])
    plot = df.plot.pie(subplots=True, figsize=(11, 6))
    return plot

plot = plotpie(vc_real)
plot = plotpie(vc_prev)'''

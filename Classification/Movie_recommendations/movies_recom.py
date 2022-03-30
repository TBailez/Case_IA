

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
#IMPORT DAS BASES

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tag = pd.read_csv('tags.csv')
notlisted = movies[movies['genres']=='(no genres listed)']
movies = movies[movies['genres']!='(no genres listed)']
'''
movies.info()
ratings.info()
tag.info()
'''
#%%
#VER OS FILMES MAIS AVALIADOS
rated = pd.DataFrame()
tagged = pd.DataFrame()

    
def m_rated(df):
    most_rated = ratings['movieId'].value_counts()
    for i in most_rated.index:
        df = df.append(movies[movies['movieId']==i])
    most_rated = ratings['movieId'].value_counts()
    most_rated = pd.DataFrame(most_rated)
    most_rated = most_rated.rename(columns={'movieId':'n_rates'})
    most_rated['movieId'] = most_rated.index
    most_rated = most_rated.sort_values('movieId')
    return df,most_rated
def m_tagged(df):
    most_tagged = tag['movieId'].value_counts()
    for i in most_tagged.index:
        df = df.append(movies[movies['movieId']==i])
    most_tagged = tag['movieId'].value_counts()
    most_tagged = pd.DataFrame(most_tagged)
    most_tagged = most_tagged.rename(columns={'movieId':'n_tags'})
    most_tagged['movieId'] = most_tagged.index
    most_tagged = most_tagged.sort_values('movieId')
    return df, most_tagged


rated, most_rated = m_rated(rated)
tagged, most_tagged = m_tagged(tagged)


movies = movies.merge(most_rated,on = 'movieId',how='left',validate='many_to_one')
#movies = movies.merge(most_tagged,on = 'movieId',how='left',validate='many_to_one')
#movies = movies.fillna(0)
#%%
#MONTAR VARIAS COLUNAS COM  GENEROS

genres = pd.DataFrame(columns=['genre1', 'genre2', 'genre3', 'genre4','genre5'])
def genero(df,dff):
    for i in df['genres']:
        y = i.split('|')
        if len(y)<5:
            to_5 = 5 - len(y)
            c=0
            while to_5>0:
                y.append(y[c])
                to_5-=1
        elif len(y)>5:
            y = y[0:5]  
        length = len(dff)
        dff.loc[length] = y
    return dff




try:
    genres = genero(movies,genres)
    movies = movies.join(genres)
    del movies['genres']
    media_grouped = ratings.groupby(['movieId']).mean()
    media_ratings = pd.DataFrame(media_grouped['rating'])
    media_ratings.reset_index(inplace=True)
    movies = pd.merge(movies, media_ratings,how = 'left',on= 'movieId')
    '''
    median_grouped = ratings.groupby(['movieId']).median()
    median_ratings = pd.DataFrame(median_grouped['rating'])
    median_ratings.rename(columns = {'rating':'rating_median'}, inplace = True)
    median_ratings.reset_index(inplace=True)
    movies = pd.merge(movies, median_ratings,how = 'left',on= 'movieId')'''
    
except:
    print('Ja rodou essa celula')
    pass


#%%
no_ratings = movies[movies.isna().any(axis=1)]
#movies.info()
films = movies.dropna()
title = films.pop('title')
#films.info()
#%%
#Rodar label  encoder para rransformar string em numero
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
multi = MultiColumnLabelEncoder(columns=['genre1', 'genre2', 'genre3', 'genre4','genre5'])

df  = multi.fit_transform(films)

#cor = df.corr()
#%%
###############################################################################################
#MONTAR SCALER
scaler = MaxAbsScaler()

#MONTAR MODELO NMF
nmf = NMF()

#NORMALIZADOR
normalizer = Normalizer()

    #MONTAR A PIPELINE
pipeline = make_pipeline(scaler,normalizer,nmf)


ppl = pipeline.fit_transform(df)
dff = pd.DataFrame(ppl,index=films['movieId'])

def select_movie():
    print('Escreva parte do nome do filme:')
    movie_name = input()
    x = title.str.contains(movie_name, regex=False)
    x = pd.DataFrame(x)
    
    result_df= x.loc[x['title']==True].index
    
    if len(result_df) == 0:
        print('Esse filme não está na base')
    else:
        c = 0
        ll = []
        print(len(result_df), 'Resultado(s) para busca:')
        for i in result_df:
            y = movies.loc[[i]][['title']]
            ll.append(y['title'].iloc[0])
            print(c, y['title'].iloc[0])    
            c +=1
        print('Selecione uma opção')  
        opc = int(input())
        print('Opção selecionada:', ll[opc])
    
    esc = title.str.contains(ll[opc], regex=False)
    esc = pd.DataFrame(esc)
    result_esc= esc.loc[esc['title']==True].index
    movie_to_rec = movies.iloc[result_esc[0]]['movieId']
    return movie_to_rec

def select():
    movie_to_rec = select_movie()
    filme_base = dff.loc[movie_to_rec]
    
    similarities = dff.dot(filme_base)
    movie_rec = similarities.nlargest(11)
    #print(similarities.nlargest(11))
    return movie_rec


def recommend():
    movie_rec = select()
    c=0
    l = movie_rec.index.tolist()
    for i in l:
        if c>0:
            r1=movies[movies['movieId']==i]
            print('Recomendação',c,':',r1.loc[r1['movieId'] == i, 'title'].values[0])
            c+=1
        else:
            c+=1
            pass
recommend()

while True:
    print('dnv? S/N')
    res = input()
    if res.upper() == 'S':
        recommend()
    elif res.upper() == 'N':
        break
    else:
        pass
        
        
        
        

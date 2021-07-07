#Importo as bibliotecas necessárias

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#%%

#Declaro o datraframe e monto uma lista com o nome das colunas para visualização

df = pd.read_csv('titanic_data.csv')
#%%
#Printo algumas insformações do dataframe para análise
    
print(df.info())
#%%
#Ao verificar que a coluna embarked somente tinha dois NaN, resolvi colocar o maior valor que tinha para substituir os NaN

print(df['Embarked'].value_counts())
df['Embarked'] = df['Embarked'].fillna('S')
#%%
#Divido os passageiros em vários agrupamentos de acordo com as idades

group = ['Newborn', 'Toddler','Child', 'Teen','Young Adult', 'Adult', 'Elder']
ages = [1, 3, 13, 19, 30, 60, 100]
ages_2 = [0, 1, 3, 13, 19, 30, 60]
for i in range(7):
    m = (df['Age'] <= ages[i]) & (df['Age'] > ages_2[i])
    df.loc[m,  'AgeGroup'] = group[i]
df['AgeGroup'] = df['AgeGroup'].fillna('Unknown')

#%%
#Plotei alguns gráficos para visualizar como cada situação em cada coluna favorecia ou não na sobrevivência do indivíduo

l = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked','AgeGroup']
for i in l:    
    sns.barplot(data = df, x = i, y = 'Survived',)
    plt.show()

#Cheguei a conclusão de que PClass, Sex, SibSp, Embarked e AgeGroup são importantes

#%%
#Criei os dataframes para treino e teste

train = df
test = df


colun_del_1 = ['PassengerId', 'Name', 'Age', 'Parch', 'Ticket','Fare', 'Cabin']
train = train.drop(colun_del_1, axis=1)

colun_del_2 = ['Name', 'Age', 'Parch', 'Ticket','Fare', 'Cabin','Age']
test = test.drop(colun_del_2, axis=1)

#%%
#Preciso transformar os strings que tenho no dataset em integers

labelencoder = LabelEncoder()

#primeiro do de train
train.iloc[:,2] = labelencoder.fit_transform(train.iloc[:, 2].values)
train.iloc[:,4] = labelencoder.fit_transform(train.iloc[:, 4].values)
train.iloc[:,5] = labelencoder.fit_transform(train.iloc[:, 5].values)

#em seguida de test
test.iloc[:,3] = labelencoder.fit_transform(test.iloc[:, 3].values)
test.iloc[:,5] = labelencoder.fit_transform(test.iloc[:, 5].values)
test.iloc[:,6] = labelencoder.fit_transform(test.iloc[:, 6].values)

#%%

X = train.iloc[:, 1:6].values
y = train.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)

#%%
neighbors = np.arange(1, 31)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop de 30 vizinhos diferentes
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
    

plt.title('k-NN: Diferentes quantidades de vizinhos')
plt.plot(neighbors, test_accuracy, label = 'Precissão de Testing')
plt.plot(neighbors, train_accuracy, label = 'Precissão de Training')
plt.legend()
plt.xlabel('Número de vizinhos')
plt.ylabel('Precissão')
plt.show()

#%%
#Vejo o número ideal de vizinhos

t_acc = test_accuracy.tolist()
max_test = max(t_acc)
nn = t_acc.index(max(t_acc))


#%%
knn = KNeighborsClassifier(n_neighbors = nn)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
acc = accuracy_score(y_predict,y_test)
print('Precisão de:', acc)

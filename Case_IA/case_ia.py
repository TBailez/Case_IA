#Importo as bibliotecas necessárias

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#%%

#Declaro o datraframe e monto uma lista com o nome das colunas para visualização]

df = pd.read_csv('titanic_data.csv')
dfc = df.columns

#Printo algumas insformações do dataframe para análise

print(df.isnull().sum())
print(df.dtypes)
print(df.info())
print(df['Survived'].value_counts())

#Ao verificar que a coluna embarked somente tinha dois NaN, resolvi colocar o maior valor que tinha para substituir os NaN

print(df['Embarked'].value_counts())
df['Embarked'] = df['Embarked'].fillna('S')

#Printei mais algumas informções sobre algumas colunas

print(df['Ticket'].value_counts())
print(df['Fare'].value_counts())
print(df['Age'].value_counts())

#Plotei alguns gráficos para visualizar como cada situação em cada coluna favorecia ou não na sobrevivência do indivíduo

l = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
for i in l:    
    sns.barplot(data = df, x = i, y = 'Survived',)
    plt.show()

#Cheguei a conclusão de que PClass, Sex, SibSp e Embarked são importantes


#Criei os dataframes para treino e teste

train = df
test = df


colun_del_1 = ['PassengerId', 'Name', 'Age', 'Parch', 'Ticket','Fare', 'Cabin']
train = train.drop(colun_del_1, axis=1)
print(train.head())

colun_del_2 = ['Name', 'Age', 'Parch', 'Ticket','Fare', 'Cabin']
test = test.drop(colun_del_2, axis=1)
print(test.head())

#%%
#Preciso transformar os strings que tenho no dataset em integers

labelencoder = LabelEncoder()

#primeiro do de train
train.iloc[:,2] = labelencoder.fit_transform(train.iloc[:, 2].values)
train.iloc[:,4] = labelencoder.fit_transform(train.iloc[:, 4].values)

#em seguida de test
test.iloc[:,2] = labelencoder.fit_transform(test.iloc[:, 2].values)
test.iloc[:,4] = labelencoder.fit_transform(test.iloc[:, 4].values)

#%%

X = train.iloc[:, 1:5].values
y = train.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#%%
neighbors = np.arange(1, 30)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

scores = []

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
    
    
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

t_acc = test_accuracy.tolist()
max_test = max(t_acc)
nn = t_acc.index(max(t_acc))

#%%

scores = []

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)


# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 08:34:57 2017

@author: LILIANE e GEZZIRRE
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


#carregando os dados
abrigo = pd.read_csv(r'C:\train.csv')

def Idadepordia(x):
    num, unit = x.split(' ')
    if unit == 'year' or unit == 'years':
        x = int(num) * 365
    elif unit == 'month' or unit == 'months':
        x = int(num) * 30
    elif unit == 'week' or unit == 'weeks':
        x = int(num) * 7
    elif unit == 'day' or unit == 'days':
        x = int(num)
    return x
    
def get_mainC(x):
    x = str(x)
    if x.find('Black') >= 0: return 'dark'
    if x.find('White') >= 0: return 'light'
    if x.find('Brown') >= 0: return 'dark'
    if x.find('Orange') >= 0: return 'light'
    if x.find('Blue') >= 0: return 'medium'
    if x.find('Red') >= 0: return 'medium'
    if x.find('Tan') >= 0: return 'medium'
    if x.find('Tortie') >= 0: return 'dark'
    if x.find('Calico') >= 0: return 'medium'
    if x.find('Torbie') >= 0: return 'medium'
    if x.find('Buff') >= 0: return 'light'
    if x.find('Sable') >= 0: return 'dark'
    if x.find('Cream') >= 0: return 'light'
    if x.find('Yellow') >= 0: return 'light'
    if x.find('Lynx') >= 0: return 'light'
    if x.find('Chocolate') >= 0: return 'dark'
    if x.find('Gray') >= 0: return 'dark'
    if x.find('Liver') >= 0: return 'dark'
    if x.find('Flame') >= 0: return 'light'
    if x.find('Agouti') >= 0: return 'dark'
    if x.find('Pink') >= 0: return 'light'
    if x.find('Ruddy') >= 0: return 'medium'
    if x.find('Gold') >= 0: return 'light'
    if x.find('Silver') >= 0: return 'light'
    if x.find('Lilac') >= 0: return 'light'
    if x.find('Seal') >= 0: return 'medium'
    if x.find('Fawn') >= 0: return 'light'
    if x.find('Apricot') >= 0: return 'light'
    else: return x

def Breed2(x):
    if '/' in x or 'Mix' in x:
        return 'Mix'
    else:
        return 'Pure'
        


############################Tratamento dos dados categoricos############################ 
print(abrigo['OutcomeType'])
abrigo['OutcomeType']=abrigo['OutcomeType'].replace("Adoption",1)
abrigo['OutcomeType']=abrigo['OutcomeType'].replace("Return_to_owner",2)
abrigo['OutcomeType']=abrigo['OutcomeType'].replace("Euthanasia",4)
abrigo['OutcomeType']=abrigo['OutcomeType'].replace("Transfer",3)
abrigo['OutcomeType']=abrigo['OutcomeType'].replace("Died",5)

abrigo.hist(column='OutcomeType',    # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20) 


y = abrigo['OutcomeType']

ax2 = pd.scatter_matrix(abrigo.iloc[:,:5], c=y, figsize=(15, 15), marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=.8)


mediana_OutcomeType = np.median([el for el in abrigo["OutcomeType"] if (np.isnan(el) == False)])
print("\nMediana o atributo OutcomeType:", mediana_OutcomeType, sep='\n')
#Retornam para o abrigo 

print(abrigo['SexuponOutcome'])
abrigo['SexuponOutcome']=abrigo['SexuponOutcome'].replace("Neutered Male",0)
abrigo['SexuponOutcome']=abrigo['SexuponOutcome'].replace("Spayed Female",1)
abrigo['SexuponOutcome']=abrigo['SexuponOutcome'].replace("Intact Male",2)
abrigo['SexuponOutcome']=abrigo['SexuponOutcome'].replace("Intact Female",3)
abrigo['SexuponOutcome']=abrigo['SexuponOutcome'].replace("Unknown",4)
print(abrigo['SexuponOutcome'])

abrigo.hist(column='SexuponOutcome',    # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20) 
mediana_SexuponOutcome= np.median([el for el in abrigo["SexuponOutcome"] if (np.isnan(el) == False)])
print("\nMediana o atributo SexuponOutcome:", mediana_SexuponOutcome, sep='\n')
########################Media do SexuponOutcome é 1.0#############################

print(abrigo['AnimalType'])
abrigo['AnimalType']=abrigo['AnimalType'].replace("Dog",0)
abrigo['AnimalType']=abrigo['AnimalType'].replace("Cat",1)
print(abrigo['AnimalType'])

abrigo.hist(column='AnimalType',    # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20) 
mediana_AnimalType= np.median([el for el in abrigo["AnimalType"] if (np.isnan(el) == False)])
print("\nMediana o atributo SexuponOutcome:", mediana_AnimalType, sep='\n')
########################Media do AnimalType 0.0 que é Dog #############################
print(abrigo['DateTime'])
abrigo['Year']= pd.to_datetime(abrigo['DateTime']).dt.year
abrigo['Month']= pd.to_datetime(abrigo['DateTime']).dt.month
abrigo['Day']= pd.to_datetime(abrigo['DateTime']).dt.day
print(abrigo['Year'])
print(abrigo['Month'])
print(abrigo['Day'])

abrigo.hist(column='Year',  # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20) 
mediana_Year= np.median([el for el in abrigo["Year"] if (np.isnan(el) == False)])
print("\nMediana o atributo Year:", mediana_Year, sep='\n')

abrigo.hist(column='Month',  # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20) 
mediana_Month= np.median([el for el in abrigo["Month"] if (np.isnan(el) == False)])
print("\nMediana o atributo Month:", mediana_Month, sep='\n')

abrigo.hist(column='Day',  # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20) 

mediana_Day= np.median([el for el in abrigo["Day"] if (np.isnan(el) == False)])
print("\nMediana o atributo Day:", mediana_Day, sep='\n')

print(abrigo['AgeuponOutcome'])
abrigo['AgeuponOutcome'] = abrigo['AgeuponOutcome'].fillna('0 day')
abrigo['AgeuponOutcome'] = abrigo['AgeuponOutcome'].apply(Idadepordia)
print(abrigo['AgeuponOutcome'])

abrigo.hist(column='AgeuponOutcome',  # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20) 

mediana_AgeuponOutcome= np.median([el for el in abrigo["Day"] if (np.isnan(el) == False)])
print("\nMediana o atributo AgeuponOutcome:", mediana_AgeuponOutcome, sep='\n')

print(abrigo['Color'])
abrigo['Color'] = abrigo['Color'].apply(get_mainC)
abrigo['Color'] = abrigo['Color'].replace("dark",0)
abrigo['Color'] = abrigo['Color'].replace("medium",1)
abrigo['Color'] = abrigo['Color'].replace("light",2)
abrigo['Color'] = abrigo['Color'].replace("Tricolor",3)
print(abrigo['Color'])

abrigo.hist(column='Color',  # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20) 

abrigo['Breed'] = abrigo['Breed'].apply(Breed2)
abrigo['Breed'] = abrigo['Breed'].replace("Mix",0)
abrigo['Breed'] = abrigo['Breed'].replace("Pure",1)
print(abrigo['Breed'])


abrigo.hist(column='Breed',  # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20) 

mediana_Breed= np.median([el for el in abrigo["Breed"] if (np.isnan(el) == False)])
print("\nMediana o atributo Breed:", mediana_Breed, sep='\n')

############################Exploração de dados############################
print("\nDimensões de abrigo:\n{0}\n".format(base.shape))
print("\nCampos de abrigo:\n{0}\n".format(base.keys()))
print("\nTipos dos dados:\n{0}\n".format(base.dtypes))
print(base.head(5))

############################Estatística descritiva dos dados############################
#descricao_abrigo = abrigo.describe()
#print(descricao_abrigo)
#categorical = abrigo.dtypes[abrigo.dtypes == "object"].index
#print("\n", categorical, sep='\n')
#print("\n", abrigo[categorical].describe(), sep='\n')

############################Exploração de atributos############################ 
##AnimalID, OutcomeType,AnimalType, SexuponOutcome, AgeuponOutcome, Breed, Color.

#print("\nAnálise do atributo AnimalID:", sorted(abrigo["AnimalID"])[0:10], sep='\n')
#rint(abrigo["AnimalID"].describe())
#print("\nAnálise do atributo Name:", abrigo["Name"].describe())
#print("\nAnálise do atributo DateTime:", sorted(abrigo["DateTime"])[0:10], sep='\n')
#print(abrigo["DateTime"].describe())
#print("\nAnálise do atributo OutcomeType:", sorted(abrigo["OutcomeType"])[0:10], sep='\n')
#print(abrigo["OutcomeType"].describe())
#print("\nAnálise do atributo OutcomeSubtype:", abrigo["OutcomeSubtype"].describe())
#print("\nAnálise do atributo AnimalType:", sorted(abrigo["AnimalType"])[0:10], sep='\n')
#print(abrigo["AnimalType"].describe())
#print("\nAnálise do atributo SexuponOutcome:", abrigo["SexuponOutcome"].describe())
#print("\nAnálise do atributo AgeuponOutcome:", abrigo["AgeuponOutcome"].describe())
#print("\nAnálise do atributo Breed:", sorted(abrigo["Breed"])[0:10], sep='\n')
#print(abrigo["Breed"].describe()) 
#print("\nAnálise do atributo Color:", sorted(abrigo["Color"])[0:10], sep='\n')
#print(abrigo["Color"].describe()) 

############################Remoção de atributos irrelevantes############################
## Nome do animal não é relevante pois o animal provavelmente mudará seu nome 
## Assim também os abritutos OutcomeSubtype e DateTime

del abrigo["Name"]
del abrigo["OutcomeSubtype"]
del abrigo["DateTime"]


############################Valores omissos############################

print(abrigo.head())
Zero_Sexo_Animal = np.where(abrigo["SexuponOutcome"].isnull(), 0, abrigo["SexuponOutcome"])
abrigo["SexuponOutcome"] = Zero_Sexo_Animal 

Zero_Idade_Animal = np.where(abrigo["AgeuponOutcome"].isnull(), 0, abrigo["AgeuponOutcome"])

abrigo["AgeuponOutcome"] = Zero_Idade_Animal 

omissos_idade = np.where(abrigo["AgeuponOutcome"].isnull() == True)
print("\nQuantidade de valores omissos no atributo Idade:", 
          len(omissos_idade[0]), sep='\n')

print(abrigo["AgeuponOutcome"].isnull() == True)

omissos_sexo = np.where(abrigo["SexuponOutcome"].isnull() == True)
print("\nQuantidade de valores omissos no atributo Sexo:", 
          len(omissos_sexo[0]), sep='\n')

print(abrigo["SexuponOutcome"].isnull() == True)


############################Detectando outliers (valores estremos) ############################

abrigo["AnimalType"].plot(kind="bar", figsize=(9,9))

index = np.where(abrigo["AnimalType"] == max(abrigo["AnimalType"]) )

print("Registros com valores extremos:",abrigo.loc[index], sep='\n')

############################Random Forest e Árvoré de decisão############################

    
base = abrigo.drop(['AnimalID'],axis=1).dropna()
                    
X_train, X_test, y_train, y_test = train_test_split(base.drop(['OutcomeType'],axis=1),
                                                    base['OutcomeType'], 
                                                    random_state=0)
clf = tree.DecisionTreeClassifier()
RFclass = RandomForestClassifier(n_estimators=100,min_samples_split=3,n_jobs=2)
modelDT = clf.fit(X_train, y_train)
modelRF = RFclass.fit(X_train, y_train)

y_predDT = modelDT.predict(X_test)
print(classification_report(y_test, y_predDT))  #metricas DecisonTree
y_predRF = modelRF.predict(X_test)
print(classification_report(y_test, y_predRF))  #metricas RandomForest
print (confusion_matrix(y_test, y_predRF))
print(accuracy_score(y_test, y_predRF))

le = LabelEncoder()
y = le.fit_transform(abrigo.iloc[:,(base.shape[1] - 1)])

class_names = le.classes_

print(classification_report(y_predDT, y_predRF))

###############################################################################
rf = RandomForestClassifier(n_estimators=500, random_state=0)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
            axis=0)
indices = np.argsort(importances)[::-1]

print ("Ranking das Funções")

for f in range(X_train.shape[1]):
    print('{}. feature {} ({})'.format(f+1, indices[f], importances[indices[f]]))
    
    plt.figure()
plt.title('Características importantes')
plt.bar(range(X_train.shape[1]), importances[indices],
       color='r', yerr=std[indices], align='center')
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()







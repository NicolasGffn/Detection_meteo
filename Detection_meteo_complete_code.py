import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt
import statistics as stat
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

	
d = pd.read_csv("weather.csv")
d0 = d.copy() # Tableau originel, sans aucune modification
del d['RISK_MM'] # Variable tendencieuse, les professeurs ont conseillé de l'enlever
d1 = d.copy()


def all_in() :
    # Lance l'ensemble du code nécessaire à l'obtention de nos précisions de prédiction
    global d1, d, d2
    d = nettoyage_tableau()
    d1 = encodeur()
    d1 = normalisation()
    d2 = correlation(0.2)
    creation_jeux(0.2)
    apprentissage()
    precision()
    return "Done"

    
    
    

def examen(Tableau,Colonne):
    dh=Tableau.copy()
    print ("Taille =",Tableau.shape)
    print (Tableau.info())
    print (Tableau.isnull().sum())
    if Colonne == 'all' :
        for col in Tableau.columns :
        # Au cas où on veut réaliser des histogrammes sur le tableau originel
            if dh[col].dtype != 'float64' :
                del dh[col]
        return pd.DataFrame.hist(dh)
    else :
        if Tableau[Colonne].dtype == 'float64' :
    # Affichage de l'histogramme choisi
            Colonne = Tableau.loc[:,Colonne]
            Colonne = pd.Series.to_frame(Colonne)
            pd.DataFrame.hist(Colonne)
        else :
            print("Impossible d'effectuer un histogramme sur une colonne de caractères")
            # Création d'un message d'erreur
        
    
    


# Etape 1 et 2    
def nettoyage_tableau(): 
    global d1
# On supprime les colonnes avec plus d'1/3 de valeurs manquantes
    L=[]
    for i in range (d.shape[1]) :
        # Colonne i
        a = d.iloc[:,i]  
        # Nombre de valeurs manquantes par colonne
        b = a.isnull().sum() 
        if b > d.shape[0]/3 :
            L.append(i)
    for k in range(1,len(L)+1) :
        del d[d.columns[L[len(L)-k]]]
    
# On transforme les valeurs manquantes par la valeur moyenne de leur colonne
    le = LabelEncoder()
    MOT=[]
    x=0
    l=0
    k=0
    i=0
    # On travaille avec une copie pour ne pas perdre la place des Nan
    di = d.copy()
    for col in d.columns :
        # isnull() détermine la présence de valeurs manquantes
        if d[col].isnull().sum() != 0:
            if d[col].dtype != 'float64' :
            # Si les colonnes sont des colonnes de caractères
                di.iloc[:,k].fillna('A', inplace = True)
                # On a remplacé tous les nan par des A, qui sera forcément le premier mot dans l'ordre alphabétique
                le.fit(di.iloc[:,k])
                T = le.transform(di.iloc[:,k])
                # T est la liste des numéros issus du LabelEncoder
                # Les valeurs 0 de T correspondent aux 'A' et donc au valeurs manquantes du DataFrame
                # On veut maintenant calculer la moyenne de la liste T, en ne prenant pas en compte les 0 
                for p in range(len(T)) :
                    if T[p] != 0 :
                        x += T[p]
                        l += 1
                m = int(round(x/l))
                # m est la moyenne arrondie par défaut, car LabelEncoder travaille avec des entiers
                MOT.append(le.inverse_transform([m])[0])
                # On stocke les mots (moyennnes passées au LabelEncoder inverse) dans une liste MOT
                m=0
                x=0
                l=0
            
            if d[col].dtype == 'float64' :
            # Si les colonnes sont des colonnes de flottants
            # On en profite pour modifier les valeurs aberrantes
            # Ici la condition choisie est : si une valeur est supérieure à 2.5 fois la moyenne (condition arbitraire)
                M = d[col].mean()
                N = pd.Series.to_list(d[col])
                for p in range (len(N)) :
                    if abs(N[p]) > (2.5*M) :
                        N[p] = M
                # Passage d'une liste à une Série 
                d[col] = pd.Series(N) 
                # On recalcule la 'vraie' moyenne (sans les valeurs aberrantes)
                M = d[col].mean()
                d[col].fillna(value = M, inplace = True)
        k += 1
       
    # On remplace maintenant les valeurs manquantes des colonnes de caractères du DataFrame
    k=0
    for col in d.columns :
        if d[col].isnull().sum() != 0:
           d.iloc[:,k].fillna(str(MOT[i]), inplace = True)
           i+=1
        k += 1
    
    d1 = d.copy() # On réactualise d1
    
    return d



# Etape 3
def encodeur():
    # Retourne un tableau d1 avec les valeurs littérales encodées en nombres
    le = LabelEncoder()
    for col in d.columns :
        if d[col].dtype != 'float64' :
            le.fit(d[col])
            T = le.transform(d[col])
            d1[col] = T
    return d1
  
         
def normalisation():
    scaler = StandardScaler()
    for col in d1.columns :
        D = d1[col]
        # StandardScaler() prend des 2D array en arguments 
        D = np.array(D).reshape(-1, 1) 
        scaler.fit(D)
        D = scaler.transform(D)
        d1[col] = D
    return d1
   
    

# Etape 4       
def correlation(r):
    # On ne veut étudier que les colonnes qui ont une certaine corrélation avec RainTomorrow
    L=[]
    global d2
    d2 = d1.corr()

    # Suppression des colonnes inutiles, celles dont la corrélation avec RainTomorrow n'est pas assez forte 
    for i in range(d2.shape[0]) :
        if abs(d2.loc[d2.index[i],'RainTomorrow']) < r :
            L.append(i)
    for p in range(1,len(L)+1) :
        del d2[d2.columns[L[len(L)-p]]]
    L=[]  
        
    # Suppression des lignes inutiles (juste pour une meilleure visibilité)       
    for o in range(d2.shape[0]) :
        if d2.index[o] != 'RainTomorrow' : 
            L.append(o)
    for u in range(1,len(L)+1) :
        d2.drop([d2.index[L[len(L)-u]]], inplace = True)
    # d2 permet l'affichage des coefficients de corrélation de manière très visible
    
    global d3
    d3 = d1.copy()
    for col in d1.columns :
        if col not in d2.columns :
            del d3[col]
    # On ne veut regarder que les colonnes intéressantes
    # d3 est le DataFrame contenant uniquement les colonnes intéressantes
    return d2



def nuages_de_points():
    pd.plotting.scatter_matrix(d3)
    
    

# Etape 5
def creation_jeux(x):
    global y,d4
    y = d1['RainTomorrow'].values  # C'est la colonne que l'on veut prédire avec le modèle
    d4 = d3.copy()
    # d4 va être le DataFrame du jeu de données test
    del d4['RainTomorrow'] # On la supprime des données servant à l'apprentissage
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(d4.values, y, test_size = x, random_state=32)
    # Les jeux de données sont créés
    return X_train, X_test, y_train, y_test



# Etape 6
def apprentissage() :
    global y_pred
    logreg = LogisticRegression()
    # On passe ensuite en binaire pour y_train
    for i in range (y_train.shape[0]):
        y_train[i]=int(y_train[i])
    for i in range (y_test.shape[0]):
        y_test[i]=int(y_test[i])  
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    return 'Done' 



# Etape 7
def precision() :
    print ('Precision score =', precision_score(y_test, y_pred, average = 'weighted'))
    print ('Accuracy score =', accuracy_score(y_test, y_pred))
    print('Recall score =', recall_score(y_test, y_pred, average = 'weighted'))
    print ('F1 score =', f1_score(y_test, y_pred, average = 'weighted'))
    
    


def confusion() :
    print('Matrice de confusion :')
    print(confusion_matrix(y_test, y_pred))
    


# Etape 8
def cross_validation(k):
    # Passage en binaire pour la sortie 
    for i in range (y.shape[0]):
        y[i]=int(y[i])
    
    # Découpage en k Folds
    kfold = model_selection.KFold(n_splits=k, random_state=7)
    
    # Réalisation de la cross validation
    model = LogisticRegression()
    results = model_selection.cross_val_score(model, d4.values, y, cv=kfold)
    
    # Affichage des statistiques
    print("Données sur l'ensemble des précisions :")
    print("Moyenne :", results.mean())
    print("Variance :", np.var(results))
    print("Ecart-type :", stat.stdev(results))
    results = plt.hist(results, bins='auto')
    plt.title("Histogramme des k validations croisées")
    plt.show()
    print('Valeur de k :', k)




# Etape 9
start_time = time.time()
all_in()
print("Temps d'éxécution : %s secondes" % (time.time() - start_time))

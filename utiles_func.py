from sklearn import linear_model
import matplotlib.pyplot as plt
import math
from tqdm import tqdm 
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split




def single_linear_regression(x,y,col=0):
    """
    prend x et y comme variable numerique en format numpy
    return reg.coef_, reg.intercept_ and figure (model fit inside data points. choose col to plot it with y)
    """
    if x.ndim==1:
        x=x.reshape(x.shape[0],1)
    reg = linear_model.LinearRegression()
    reg.fit(x,y)
    #
    plt.scatter(x[:,col],y,color='red',marker='+')
    plt.plot(x, reg.predict(x))
    export_model(reg)
    return reg.coef_, reg.intercept_


def gradient_descent(x,y, learning_rate = 0.0002, iterations = 1000000, export=True):
    """
    prend x et y comme variable numerique en format numpy
    return m and b (coef and bias)
    """
    m_curr = b_curr = 0
    n = len(x)
    cost_previous = 0
    history = pd.DataFrame(index=range(iterations), columns=['m','b','Cost'])

    for i in tqdm(range(iterations)):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost,cost_previous,rel_tol=1e-20):
            break
        cost_previous = cost
        history.iloc[i,:] = [m_curr,b_curr,cost]
        #print(f"m {m_curr}, b {b_curr}, Cost {cost}, Iteration {i}")

    if export:
        history.to_csv('history.csv')
    return m_curr, b_curr


def export_model(model):
    """
    Export your model as pickle file (binary file)
    """
    with open ('model_pickle', 'wb') as f:
        pickle.dump(model,f)
    print('Your model has been succ exported !')


def encode_categorical_and_split(df, test_size=0.2, random_state=None):
    """
    Transforme toutes les variables catégorielles en dummies, supprime les colonnes d'origine,
    et ajoute une colonne 'Set' pour identifier les lignes du train et du test.

    Parameters:
    df (pd.DataFrame): Le DataFrame contenant des variables catégorielles et numériques.
    test_size (float): La proportion des données à mettre dans le test (ex: 0.2 pour 20%).
    random_state (int, optional): La graine aléatoire pour la reproductibilité.

    Returns:
    pd.DataFrame: Un DataFrame avec les variables catégorielles encodées et la colonne 'Set'.
    """
    # Identifier les variables catégorielles
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Encoder en dummies et supprimer les colonnes d'origine
    dummies = pd.get_dummies(df[categorical_cols], dtype='int')#, drop_first=True)
    df_encoded = pd.concat([df.drop(columns=categorical_cols), dummies], axis=1)
    
    # Séparer en train et test
    train_idx, test_idx = train_test_split(df_encoded.index, test_size=test_size, random_state=random_state)
    
    # Ajouter la colonne 'Set'
    df_encoded["Set"] = "Train"
    df_encoded.loc[test_idx, "Set"] = "Test"
    
    return df_encoded
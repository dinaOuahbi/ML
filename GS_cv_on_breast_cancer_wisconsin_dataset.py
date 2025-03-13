# %% [markdown]
# ## Grid search CV

# %% [markdown]
# # Les hyperparamètres dans l'apprentissage automatique
# sont des éléments configurables avant l'entraînement d'un modèle qui influencent son comportement et ses performances.
# 
# # GridSearchCV
# tool for identifying the optimal parameters for a machine learning model.
# You provide GridSearchCV with a set of Scoring parameter to experiment with
# evaluates the model’s performance : on various sections of the dataset
# After, GridSearchCV presents you with the combination of settings that yielded the most favorable outcomes.
# 
# Grid Search employs an exhaustive search strategy, This approach involves tuning parameters, such as learning rate
# when you have a larger dataset, go with RandomizedSearchCV and not GridSearchCV.
# course : https://towardsdatascience.com/gridsearchcv-for-beginners-db48a90114ee/
# 
# # HalvingGridSearchCV
# Utilise une approche successive halving, où il commence avec un grand nombre de combinaisons mais élimine progressivement les moins performantes à chaque itération, ce qui réduit considérablement le temps de calcul.
# 
# # Bayesian Optimization
# is an optimization method that uses probabilistic models to efficiently find a model’s hyperparameters using : 
# - Le modèle de substitution: comment les différentes valeurs des hyperparamètres affectent les performances du modèle.
# - La fonction d'acquisition est une mesure mathématique qui évalue l'intérêt (ou l'utilité) d'évaluer une configuration d'hyperparamètres donnée
# 
# - La stratégie d’équilibre entre exploration et exploitation est l’approche utilisée pour décider s’il faut explorer de nouvelles configurations d’hyperparamètres pour découvrir des améliorations potentielles (exploration) ou exploiter les configurations actuellement connues considérées comme les meilleures (exploitation).
# (https://inside-machinelearning.com/en/bayesian-optimization/)
# 
# 
# 

# %%
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('data/breast-cancer-wisconsin-data_data.csv')
df.drop(['Unnamed: 32','id'],axis=1,inplace=True)
df.diagnosis=[1 if each=="M" else 0 for each in df.diagnosis]
y=df.loc[:,"diagnosis"]
X=df.loc[:,df.columns!="diagnosis"]

scaler = StandardScaler()
#X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size = 0.2, random_state = 122)
print(y_train.value_counts()/X_train.shape[0])

# %%
#default
model = svm.SVC(kernel='rbf',C=30,gamma='auto')
model.fit(X_train,y_train)
print(model.score(X_test, y_test))

# %%
# svm rf naive bayes lr kmeans dt xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

model_params = {
    'svm': {
        'model': SVC(gamma='auto',probability=True),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 5, 10],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C': [1, 5, 10],
            'penalty': ['l1', 'l2']
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 50],
            'metric': ['minkowski', 'euclidean', 'manhattan'],
            'p': [1, 2]
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    },
    'xgboost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
}


# %%
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# %%
scores = []
y_score = []

for model_name, mp in tqdm(model_params.items()):
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False) # return Mean cross-validated score of the best_estimator
    clf.fit(X, y)
    y_score.append({
        'model' : model_name,
        'pred' : clf.best_estimator_.predict_proba(X)[:, 1]
        })

    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    


# %%
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df

# %%
"""pred_df = pd.DataFrame(y_score)
pred_df.set_index('model', inplace=True)
pred_df"""

# %%
"""from sklearn.metrics import roc_curve, auc

for i in pred_df.index:
    y_score = pred_df.loc[i,'pred']
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()"""

# %%
df.head()

# %%
df.loc[len(df)] = {"model":"default_svm","best_score":model.score(X_test, y_test),"best_params":"kernel=rbf,C=30,gamma=auto"}
df

# %%
#df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

sns.barplot(x='best_score', y='model', data=df, palette='viridis')

plt.xlabel("Score")
plt.ylabel("models")
plt.title("Comparaison des performances des modèles")
plt.savefig('issues/GS_cv.png')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table_data = df.values.tolist()
table_data.insert(0, df.columns.to_list())  # Ajouter les en-têtes

table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0, 1, 2])

plt.title("GridSearchCV_results")
plt.savefig('issues/hyperParams.png')
plt.show()

# %%
from sklearn.metrics import roc_curve, auc
# Fonction pour tracer les courbes ROC
def plot_roc_curve(models_params, X_train, y_train):
    plt.figure(figsize=(10, 8))

    for model_name, model_info in tqdm(models_params.items()):
        model = model_info['model']
        params = model_info['params']

        # Appliquer GridSearchCV pour chaque modèle
        grid_search = GridSearchCV(model, params, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Obtenir les probabilités prédites pour les courbes ROC
        y_score = grid_search.best_estimator_.predict_proba(X_train)[:, 1]  # Probabilités pour la classe positive

        # Calculer la courbe ROC
        fpr, tpr, _ = roc_curve(y_train, y_score)
        roc_auc = auc(fpr, tpr)  # AUC (Area Under Curve)

        # Tracer la courbe ROC
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Tracer la diagonale (aucune performance)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# %%
plot_roc_curve(model_params, X, y)



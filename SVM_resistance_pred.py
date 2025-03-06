# %% [markdown]
# # Machine learning algorithms for identifying antibiotic resistant bacteria

# %% [markdown]
# - We will be focussing on a species called Neisseria gonorrhoeae, the bacteria which cause gonorrhoea
# - Many people who are infected (especially women) experience no symptoms, helping the disease to spread
# - Resistance of these bacteria to antibiotics is rising over time, making infections hard to treat.
# - Currently in the UK, patients are only treated with ceftriaxone.
# - In this notebook, we will look at machine learning algorithms for predicting resistance to azithromycin.

# %%
# set up environment
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
import seaborn as sns
import time

# %% [markdown]
# ### Data
# - We have genome sequence and antibiotic resistance data gathered from different publicly available sources (https://microreact.org/project/N_gonorrhoeae)
# 
# - We're using unitigs, short stretches of DNA shared by a subset of the strains in our study.
# - Unitigs are an efficient but flexible way of representing DNA variation in bacteria (https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1007758)
# 
# 
# - The full dataset consists of 584,362 unitigs, which takes a long time to train models on, so for this exercise we will be using a set that has been filtered for unitigs statistically associated with resistance.
# 
# - We call the different resistance patterns 'phenotypes', which is a general term for traits that an organism has.
# 
# - Info that we need : 
#     - Presence/absence pattern of the unitigs across each sample (X) 
#     - How resistant each sample is to an antibiotic (pheno)
# 

# %%
# a function for preparing our training and testing data
def prep_data(phenotype) :
    pheno = pd.read_csv('data/gono-unitigs/metadata.csv', index_col=0)
    pheno = pheno.dropna(subset=[phenotype]) # drop samples that don't have a value for our chosen resistance profile
    pheno = pheno[phenotype]
        
    # read in unitig data
    X = pd.read_csv('data/gono-unitigs/' + phenotype + '_gwas_filtered_unitigs.Rtab', sep=" ", index_col=0, low_memory=False)
    X = X.transpose()
    X = X[X.index.isin(pheno.index)] # only keep rows with a resistance measure
    pheno = pheno[pheno.index.isin(X.index)]
    return X, pheno

# %%
# let's predict azithromycin resistance 
# prepare our data for predicting antibiotic resistance
phenotype = 'azm_sr'
X, pheno = prep_data(phenotype)

# create an array for storing performance metrics
performance = []
method = []
times = []

# %%
# take a look at the data

print(pheno.shape)
print(pheno.head())
print('-'*100)
print(X.shape)
print(X.iloc[:10,:10])

# %%
# look at the length distribution of the unitigs in our dataset
unitigs = X.columns
print(unitigs[:10])
mylen = np.vectorize(len)
uni_len = mylen(unitigs)
sns.distplot(uni_len)

# %% [markdown]
# ### Modeling
# 
# - Here we will use cross validaion to see how well the model will work on new data
# - Here is the steps : 
#     - split data into 5 folds / and buit 5 models
#     - model tuning with GridSearchCV
#     - evaluate model based on Accuracy (cv)
#     

# %%
# function for fitting a model
def fitmodel(X, pheno, estimator, parameters, modelname, method, performance, times) :
    
    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(X, pheno):
        # time how long it takes to train each model type
        start = time.process_time()
        
        # split data into train/test sets
        X_train = X.iloc[train_index]
        y_train = pheno[train_index]
        X_test = X.iloc[test_index]
        y_test = pheno[test_index]
        
        # perform grid search to identify best hyper-parameters
        gs_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=3, n_jobs=-1, scoring='balanced_accuracy')
        gs_clf.fit(X_train, y_train)
        
        # predict resistance in test set
        y_pred = gs_clf.predict(X_test)
                
        # call all samples with a predicted value less than or equal to 0.5 as sensitive to the antibiotic, 
        # and samples with predicted value >0.5 resistant to the antibiotic
        y_pred[y_pred<=0.5] = 0
        y_pred[y_pred>0.5] = 1

        score = balanced_accuracy_score(y_test, y_pred)
        performance = np.append(performance, score)
        method = np.append(method, modelname)
        times = np.append(times, (time.process_time() - start))

        print("Best hyperparameters for this fold")
        print(gs_clf.best_params_)
        print("Confusion matrix for this fold")
        print(confusion_matrix(y_test, y_pred))
    return gs_clf, method, performance, times

# %%
def sbplot(X, pheno, estimator, parameters, modelname, method, performance, times) :
    results = []
    
    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(X, pheno):
        # time how long it takes to train each model type
        start = time.process_time()
        
        # split data into train/test sets
        X_train = X.iloc[train_index]
        y_train = pheno[train_index]
        X_test = X.iloc[test_index]
        y_test = pheno[test_index]
        
        # perform grid search to identify best hyper-parameters
        gs_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=3, n_jobs=-1, scoring='balanced_accuracy')
        gs_clf.fit(X_train, y_train)
        
        # predict resistance in test set
        y_pred = gs_clf.predict(X_test)
        
        results.append([y_test, y_pred])
        
    return results

# %%
enet = SGDClassifier(loss="log_loss", penalty="elasticnet")
enet_params = {
    'l1_ratio': [0.1, 0.2, 0.5]
}

enet_model, method, performance, times = fitmodel(X, pheno, enet, enet_params, "Elastic net", method, performance, times)

# %%
svm = SVC(class_weight='balanced')
svm_params = {
    'C': [0.01],
    'gamma': [1e-06, 1e-05],
    'kernel': ['linear']
}

svm_model, method, performance, times = fitmodel(X, pheno, svm, svm_params, "Support vector machine", method, performance, times)

# %%
xgb_mod = xgb.XGBClassifier(random_state=0)
xgb_params = {
    'alpha': [1e-5, 1e-4], 
    'colsample_bytree': [0.6],
    'gamma': [0.05, 0.1], 
    'learning_rate': [0.01, 0.1], 
    'max_depth': [2], 
    'objective': ['binary:hinge'], 
    'subsample': [0.2, 0.4, 0.6]
}

xgb_model, method, performance, times = fitmodel(X, pheno, xgb_mod, xgb_params, "XGBoost", method, performance, times)

# %%
rf = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced")
rf_params = {
    'max_features': [round(X.shape[1]*0.1), round(X.shape[1]*0.5), round(X.shape[1]*0.8)],
    'max_depth': [3],
    'n_estimators': [50]
}

rf_model, method, performance, times = fitmodel(X, pheno, rf, rf_params, "Random forest", method, performance, times)

# %%
# compare results from the different predictors
sns.set_context("talk")
plt.title("Model Performance - Azithromycin Resistance", y=1.08)
sns.swarmplot(x=method, y=performance, palette="YlGnBu_d", size=10, edgecolor='black')
sns.despine()
plt.ylabel("Balanced accuracy")
plt.xticks(rotation=30, ha='right', fontsize=7)
plt.yticks(fontsize=7)
plt.savefig('issues/svm_acc.png')

# %%
# took at the time taken to train the different models
sns.set_context("talk")
plt.title("Model Training Times - Azithromycin Resistance", y=1.08)
sns.swarmplot(x=method, y=times, palette="YlGnBu_d", size=10)
sns.despine()
plt.ylabel("Time taken for training")
plt.xticks(rotation=30, ha='right')

# %% [markdown]
# ### Exploring what the model has learned

# %%
# function for looking at SVM feature importance
def plot_coefficients(classifier, feature_names, top_features=5, export=True):
    coef = classifier.best_estimator_.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    sns.set_context("poster")
    plt.figure(figsize=(10, 5))
    plt.title("Feature Importances (Support Vector Machine) - Azithromycin Resistance", fontsize=15)
    colors = ['crimson' if c < 0 else 'cornflowerblue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors, edgecolor='black', width=0.6)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    if export:
        plt.savefig('issues/FI_svm_azm_resistance.png',dpi=300)
    plt.show()
    np.asarray(feature_names)[top_positive_coefficients]


# %%
plot_coefficients(svm_model, list(X.columns))
    
# if we print the unitigs, we can then look at what genes they relate to
coef = svm_model.best_estimator_.coef_.ravel()
feature_names = list(X.columns)
top_negative_coefficients = np.argsort(coef)[:5]
print("Top negative predictors: ", np.asarray(feature_names)[top_negative_coefficients])

top_positive_coefficients = np.argsort(coef)[-5:]
print("Top positive predictors: ", np.asarray(feature_names)[top_positive_coefficients])



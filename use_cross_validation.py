# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, precision_score
from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt

# some hyper parameters
SEED = 1970
test_train_split_SEED = 1970
# FOLDS = 10
show_fold_stats = True
VERBOSE = 0
FOLDS = 5

# %% [markdown]
# Autistic Spectrum Disorder (ASD) is a severe neurodevelopmental disorder 
# abnormal behavioral symptoms
# prevalence increasing dramatically over the past decades
# 
# gastrointestinal (GI) symptoms, such as gaseousness, diarrhea, and constipation, often co-occurred with ASD core symptoms in children with ASD. 
# 
# changes in gut microbiota can modulate the gastrointestinal physiology, immune function, and even behavior through the gut-microbiome-brain axis.
# 
# 
# The dataset is from the research paper by Zhou Dan et al. published on April 21st of 2020
# https://www.tandfonline.com/doi/full/10.1080/19490976.2020.1747329#abstract
# 
# 
# 
# 
# 
# 

# %%
pd_abundance = pd.read_csv('data/ASD/GSE113690_Autism_16S_rRNA_OTU_assignment_and_abundance.csv')
pd_meta_abundance = pd.read_csv('data/ASD/ASD meta abundance.csv')

# %%
taxa = pd_abundance[['OTU', 'taxonomy']].set_index('OTU')
pd_abundance_T = pd_abundance.drop('taxonomy', axis=1).set_index('OTU').transpose()

# %%
print(taxa.shape)
taxa.head()

# %%
print(pd_abundance_T.shape)
pd_abundance_T.head()

# %%
target = pd_abundance_T.index.to_list()
binary_target = np.array([1 if t.startswith('A') else 0 for t in target ])

# %%
total_species = pd_abundance_T.sum(axis = 1)
abs_abundance = total_species.unique()[0]
pd_rel_abundance = pd_abundance_T / abs_abundance 

# %%
pd_rel_abundance.head()

# %%
X_train, X_val, y_train, y_val = train_test_split(pd_rel_abundance, binary_target, test_size=0.05, random_state=test_train_split_SEED,shuffle = True)

#model = RandomForestClassifier(n_estimators = 500, random_state = SEED, verbose = 0)
model = XGBClassifier(n_estimators=5000, max_depth=None, 
                        learning_rate=0.005,
                        objective='binary:logistic', 
                        metric='auc',
                        verbosity  = VERBOSE,
                        # tree_method = 'gpu_hist',
                        use_label_encoder=False,
                        n_jobs=-1, random_state  = SEED )


# Entraînement sans cross-validation
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]  # Probabilités pour l'AUC-ROC

# Calcul des métriques sans cross-validation
auc_no_cv = roc_auc_score(y_val, y_proba)
f1_no_cv = f1_score(y_val, y_pred)
recall_no_cv = recall_score(y_val, y_pred)
precision_no_cv = precision_score(y_val, y_pred)


# %%
# Lets put aside a small test set, so we can check performance of different classifiers against it
disease_train, disease_test, disease_y_train, disease_y_test = train_test_split(pd_rel_abundance, binary_target,
                                                                                test_size = 0.05,
                                                                                random_state = test_train_split_SEED ,
                                                                                shuffle = True)   

# %%
def get_score(clf, X_train, y_train, X_val, y_val):
    clf.fit(X_train, y_train )

    pred = clf.predict(X_val)
    pred_proba = clf.predict_proba(X_val)
    
    roc_auc = roc_auc_score(y_val, pred_proba[:,1])
    f1 = f1_score(y_val, pred)
    recall = recall_score(y_val, pred)
    precision = precision_score(y_val, pred)

    return roc_auc, f1, recall, precision, pred

# %%
skf = StratifiedKFold(n_splits = FOLDS, shuffle = True, random_state = SEED)
auc_cv, f1_cv, recall_cv, precision_cv = [[] for i in range(4)]

for fold, (idxT,idxV) in enumerate(skf.split(disease_train, disease_y_train)):

    X_train = disease_train.iloc[idxT]
    X_val = disease_train.iloc[idxV]
    y_train = disease_y_train[idxT]
    y_val = disease_y_train[idxV]

    roc_auc, f1, recall, precision, pred = get_score(model, X_train, y_train, X_val, y_val)
    auc_cv.append(roc_auc)
    f1_cv.append(f1)
    recall_cv.append(recall)
    precision_cv.append(precision)


    if show_fold_stats:
        print('-' * 80)
        print(confusion_matrix(y_val, pred))



# %%
# Labels des métriques
labels = ["AUC-ROC", "F1-score", "Recall", "Precision"]
scores_no_cv = [auc_no_cv, f1_no_cv, recall_no_cv, precision_no_cv]
scores_cv = [np.mean(auc_cv), np.mean(f1_cv), np.mean(recall_cv), np.mean(precision_cv)]

# Définition des positions
x = np.arange(len(labels))
width = 0.35

# Couleurs et style
sns.set_style("whitegrid")
palette = sns.color_palette("mako", 2)

fig, ax = plt.subplots(figsize=(8, 5))

width = 0.3  # Réduire légèrement la largeur des barres
spacing = 0.05  # Ajouter un petit espace entre les groupes de barres

bars1 = ax.bar(x - width/2 - spacing/2, scores_no_cv, width, label="Without CV", color=palette[0], edgecolor="black", alpha=0.8)
bars2 = ax.bar(x + width/2 + spacing/2, scores_cv, width, label="With CV", color=palette[1], edgecolor="black", alpha=0.8)

# Ajout des valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{bar.get_height():.2f}", 
                ha='center', fontsize=11, fontweight='bold')

# Personnalisation du graphique
ax.set_ylabel("Score", fontsize=12, fontweight='bold')
ax.set_title("Performance comparison with and without Cross-Validation", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(frameon=True, fontsize=11)

# Affichage
plt.tight_layout()
plt.savefig('issues/perf_comparaison_cv.png', dpi=200)
plt.show()




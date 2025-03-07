# %% [markdown]
# # Cirrhosis classification
# 
# - Using Shotgun metagenomics for disease prediction
# - sequence the genomes of untargeted cells in a community in order to elucidate community composition and function.
# 
# ### Methods
# - classification issue
# - response : cirrhosis against healthy samples
# - first of all I approach prediction cirrhosis using XGBoost 
# 
# - compare with RF here (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004977) by using **the reduced dataset for ongoing experimentation.**
# 
# 

# %%
from xgboost import XGBClassifier
import xgboost as xgb
import os
import sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score

import seaborn as sns
#import scikitplot as skplt
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Metaparameters
VERBOSE = 0
FOLDS = 5

show_fold_stats = True
# show_fold_stats = False # set to True if all OOF results wanted

do_plot_ROC = False # set to True to plot ROC when predicting cirrhosis

# test_train_split_SEED = 1970
test_train_split_SEED = 1971

# %%
def plot_ROC(fpr, tpr, m_name):
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc, alpha=0.5)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC for %s'%m_name, fontsize=20)
    plt.legend(loc="lower right", fontsize=16)
    plt.show()

# %%
pd_abundance = pd.read_csv('data/Metagenomic_segata_2016/abundance_stoolsubset.txt', dtype=str, sep='\t')
pd_abundance

# %%
pd_abundance = pd.read_csv('data/Metagenomic_segata_2016/abundance_stoolsubset.csv', dtype=str)
disease = pd_abundance.loc[:,'disease'] 
d_name = pd_abundance.loc[:,'dataset_name'] 
print(disease.value_counts())
print('-'*100)
print(d_name.value_counts())

# list of diseases we want to analyze and predict
diseases = ['obesity', 'cirrhosis', 't2d', 'cancer']

# %%
cols = pd_abundance.columns.tolist()

# separate data from metadata
species = [x for x in cols if x.startswith('k_')]
metadata = [x for x in cols if not x.startswith('k_')]

pd_abundance_conv = pd_abundance.copy()
pd_abundance_conv = pd_abundance_conv[species].astype('float64')
pd_abundance_conv = pd.concat([pd_abundance[metadata], pd_abundance_conv], axis = 1)

# %%
pd_abundance_conv.head()

# %%
# controls/healthy samples from Human Microbiome Project coded 'hmp' and 'hmpii'. 
# 't2d' stands for Type 2 Diabetes. We will combine a few studies into single dataset.
data_sets = {'control':['hmp', 'hmpii'],'t2d':['WT2D','t2dmeta_long','t2dmeta_short'], 'cirrhosis' : ['Quin_gut_liver_cirrhosis'], 
             'cancer' : ['Zeller_fecal_colorectal_cancer'], 'obesity' : ['Chatelier_gut_obesity']}
# combine controls from different studies into one
pd_abundance_conv['disease'] = pd_abundance_conv['disease'].apply(lambda x: 'control' if ((x == 'n') or (x == 'nd') or (x == 'leaness')) else x)

# %%
pd_abundance_conv['disease'].value_counts()

# %%
# separate controls and diseases into 2 dataframes
pd_control = pd_abundance_conv.loc[pd_abundance_conv['disease'] == 'control']
pd_disease = pd_abundance_conv.loc[pd_abundance_conv['disease'] != 'control']

print(
    pd_control.shape,
    pd_disease.shape,
)

print(pd_disease.disease.unique())

# %%
# we won't consider diseases from this list
not_disease = [d for d in pd_disease.disease.unique().tolist() if d not in diseases] 
for d in not_disease:
    pd_disease = pd_disease.drop(pd_disease.loc[pd_disease['disease'] == d].index, axis = 0)  

print(
    pd_control.shape,
    pd_disease.shape,
)  

# %%
ds_names = data_sets['cirrhosis']
pd_cont = pd_control.loc[pd_control['dataset_name'] == ds_names[0]]
pd_dis = pd_disease.loc[pd_disease['dataset_name'] == ds_names[0]]

print(
    pd_cont.shape,
    pd_dis.shape
)

# %%
merge = pd.concat([
    pd_cont,
    pd_dis
],axis=0).set_index('sampleID')

merge.head()

# %%
species = merge.select_dtypes('float').columns

# %%

X = merge.select_dtypes('float').values
y = merge['disease'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
len(X)

# %%
from sklearn.ensemble import RandomForestClassifier # for multiple algorithm (that why is called ensemble)
model = RandomForestClassifier(n_estimators=25)
model.fit(X_train, y_train)


# %%
feature_scores[:100]

# %%
feature_scores = pd.Series(model.feature_importances_, index=species).sort_values(ascending=False)[:15]

# Creating a seaborn bar plot

sns.barplot(x=feature_scores, y=feature_scores.index)
# Add labels to the graph

plt.xlabel('Feature Importance Score')
plt.ylabel('Species')
# Add title to the graph
plt.title("Visualizing Important Features")
# Visualize the graph
plt.savefig('issues/RF_feat_omportance_cirrhosis_ds.png', dpi=300)
plt.show()

# %%
model.score(X_test, y_test)

# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model.predict(X_test))

sns.heatmap(cm, annot=True)
plt.xlabel('Pred')
plt.ylabel('Truth')
plt.savefig('issues/RF_cm_cirrhosis_ds.png', dpi=300)

# %%
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(model.estimators_[2], filled=True)
plt.savefig('issues/RF_decision_tree_cirrhosis_ds.png', dpi=300)



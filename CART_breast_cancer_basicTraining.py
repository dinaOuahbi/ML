# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})

# %% [markdown]
# ### About this Dataset
# 
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree Construction Via Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], a classification method which uses linear programming to construct a decision tree. Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes. The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
# 
#         Number of instances: 569
# 
#         Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)
# 
#         Diagnosis (M = malignant, B = benign)
# 
# 
# ### Methods
# This notebook trains a basic decision tree classifier for breast cancer prediction.
#         It begins by splitting the data into training and testing sets, ensuring balanced class distribution.
#         The data is then scaled using MinMaxScaler.
# 
# A basic decision tree model is trained and evaluated, achieving reasonable precision and recall. 
# 
# 
# - Complete source for this project : https://github.com/PragyanTiwari/Breast-Cancer-Prediction-with-DecisionTree-Classifier
# 
# 

# %%
# reading the dataset
df = pd.read_csv(r"data/breast-cancer-wisconsin-data_data.csv")
df.drop(["id","Unnamed: 32"], axis=1, inplace=True)
print(f"shape: {df.shape}")
print(f"n_duplicates: {df.duplicated().sum()}")
print(f"null_values: {df.isnull().sum().sum()}")
df.head()

# %%
features = df.drop("diagnosis", axis=1)
target = df['diagnosis']

# %%
plt.figure(figsize=(8,6))

sns.countplot(x="diagnosis",
              data=df,
              palette=["#FF69B4", "#00BFFF"])
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()/len(df)*100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.title("Distribution of Classes", fontsize=18)
plt.xlabel("Class", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()

# %%
plt.figure(figsize=(5,5))
sns.clustermap(features.corr(), 
               cmap="YlGnBu", 
               figsize=(12,12), 
               method="ward",
               label="matrix")

plt.title("Features Correlation Matrix", y=1.05, fontsize=20, ha="center")
plt.show()

# %%
# importing libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# %%
X = df.drop('diagnosis',axis=1)
Y = df['diagnosis']

# train test split
x_train,x_test,y_train,y_test = train_test_split(X,Y,
                                                 test_size=0.2,
                                                 random_state=42)

print(f"(x_train: {x_train.shape}, y_train: {y_train.shape})")
print(f"(x_test: {x_test.shape}, y_test: {y_test.shape})")

# %%
# scaling down the dataset using MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train,y_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# %%
# Visualing distribution of class across the split

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# Plot distribution of y_train
sns.countplot(y_train, ax=axes[0], palette="vlag", edgecolor="black")
axes[0].set_title("Training Data")
axes[0].set_xlabel("Diagnosis (0: Benign, 1: Malignant)")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)

# Plot distribution of y_test
sns.countplot(y_test, ax=axes[1], palette="vlag", edgecolor="black")
axes[1].set_title("Testing Data")
axes[1].set_xlabel("Diagnosis (0: Benign, 1: Malignant)")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', rotation=45)


fig.suptitle("Diagnosis Distribution in Training and Testing Data")
plt.tight_layout()
plt.show()

# %%
# building a basic model

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train,y_train)


y_pred = clf.predict(x_test)

print(classification_report(y_test,y_pred))

# %%
# plotting confusion matrix

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt="d",cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion matrix: Base Decision Model')
plt.tight_layout()
plt.show()

# %%
# Exploring Feature Importance

feature_imp = {'features':X.columns,
               'importance':clf.feature_importances_}

feature_imp_df = pd.DataFrame(feature_imp)
sorted_feature_imp= feature_imp_df.sort_values(by="importance", 
                                               ascending=False)

plt.figure(figsize=(8, 8))
sns.barplot(data=sorted_feature_imp, x="importance", y="features",
            label="Feature importance",legend=True)   
plt.title("X_Train: Highest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features", fontsize=6)
plt.tight_layout()
plt.show()

# %%
features.columns

# %%
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree

plt.figure(figsize=(12, 8), dpi=300)  # Augmente la résolution
plot_tree(clf, feature_names=features.columns, class_names=y_train.unique(), filled=True, rounded=True)
# Sauvegarder l'image en haute qualité
plt.savefig("issues/CART_breast_cancer_basicTraining.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
n_least_imp_features = len(feature_imp_df.query("importance <= 0"))
print(f"total features having 0 importance: {n_least_imp_features}")

# %% [markdown]
# Atmost, 18 features are not significantly impacting the model. Instead of removing them, we'll apply PCA to create new features with reduced dimensions, preserving relevant information.
# 
# ### Perspective : 
#         Instead of discarding these features with 0 importance, w'll gona prepares the data for dimensionality reduction using PCA in a later step.
#         The training and testing sets (including features with and without importance) are saved for subsequent processing.



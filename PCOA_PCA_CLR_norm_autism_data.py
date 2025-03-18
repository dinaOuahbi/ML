# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# %%
class KNN_BrayCurtis(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        dist_matrix = cdist(X_test, self.X_train, metric='braycurtis')
        nearest_neighbors = np.argsort(dist_matrix, axis=1)[:, :self.k]
        knn_labels = self.y_train[nearest_neighbors]
        return mode(knn_labels, axis=1).mode.ravel()

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **parameters):
        for param, value in parameters.items():
            setattr(self, param, value)
        return self

# %%
pd_abundance = pd.read_csv('data/ASD/GSE113690_Autism_16S_rRNA_OTU_assignment_and_abundance.csv')
pd_meta_abundance = pd.read_csv('data/ASD/ASD meta abundance.csv')
taxa = pd_abundance[['OTU', 'taxonomy']].set_index('OTU')
pd_abundance_T = pd_abundance.drop('taxonomy', axis=1).set_index('OTU').transpose()

target = pd_abundance_T.index.to_list()
binary_target = np.array([1 if t.startswith('A') else 0 for t in target ])

total_species = pd_abundance_T.sum(axis = 1)
abs_abundance = total_species.unique()[0]
abundance_df = pd_abundance_T / abs_abundance 
print(abundance_df.shape)
abundance_df.head()


# %%
# 📌 Vérifier la distribution des abondances
sns.histplot(abundance_df.values.flatten(), bins=50, kde=True)
plt.title("Distribution des abondances relatives")
plt.show()

# %%
# 📌 Filtrer les MSPs rares (présents dans < 10% des échantillons)
min_samples = int(0.1 * abundance_df.shape[0])
filtered_df = abundance_df.loc[:, (abundance_df > 0).sum(axis=0) >= min_samples]
filtered_df.shape

# %%
# 📌 Transformation Log (évite les 0 en ajoutant un petit epsilon)
log_transformed_df = np.log(filtered_df + 1e-6)
log_transformed_df.shape

# %%
# 📌 Transformation CLR (Centered Log-Ratio)
from skbio.stats.composition import clr
clr_transformed_df = pd.DataFrame(clr(filtered_df + 1e-6), 
                                  index=filtered_df.index, 
                                  columns=filtered_df.columns)

clr_transformed_df.shape


# %%
# 📌 Affichage après transformation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(log_transformed_df.values.flatten(), bins=50, kde=True, ax=axes[0])
axes[0].set_title("Distribution après Log-transformation")

sns.histplot(clr_transformed_df.values.flatten(), bins=50, kde=True, ax=axes[1])
axes[1].set_title("Distribution après CLR-transformation")

plt.show()

# %% [markdown]
# # Étape 2 : Réduction de dimension (PCA & PCoA)

# %%
from sklearn.decomposition import PCA
from skbio.stats.ordination import pcoa
from scipy.spatial.distance import braycurtis, pdist, squareform

# %%
metadata_df = pd.DataFrame(binary_target, index=clr_transformed_df.index, columns=['Autism'])

# %%

# 📌 PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(clr_transformed_df)
pca_result.shape

# Convertir en dataframe
pca_df = pd.DataFrame(pca_result, index=clr_transformed_df.index, columns=["PC1", "PC2"])
pca_df["Autism"] = metadata_df.loc[pca_df.index, "Autism"]

# %%
# 📌 PCoA (avec distance de Bray-Curtis)
distance_matrix = squareform(pdist(clr_transformed_df, metric="braycurtis"))
pcoa_result = pcoa(distance_matrix)

# Convertir en dataframe
pcoa_df = pd.DataFrame(pcoa_result.samples.iloc[:, :2])
pcoa_df.index = clr_transformed_df.index
pcoa_df["Autism"] = metadata_df.loc[pcoa_df.index, "Autism"]

# %%
# 📌 Visualisation PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Autism", palette="viridis", alpha=0.7)
plt.title("PCA - Projection des échantillons")
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")

# 📌 Visualisation PCoA
plt.subplot(1, 2, 2)
sns.scatterplot(data=pcoa_df, x="PC1", y="PC2", hue="Autism", palette="viridis", alpha=0.7)
plt.title("PCoA - Projection des échantillons")
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")

plt.show()

# %%
# 📌 Afficher la variance expliquée pour PCA
explained_variance = pca.explained_variance_ratio_ * 100
print(f"Variance expliquée PCA : PC1 = {explained_variance[0]:.2f}%, PC2 = {explained_variance[1]:.2f}%")

# %% [markdown]
# # Étape 3 : Classification avec Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# %%
# 📌 Préparer les données pour le modèle
def prepare_data(df):
    X = df.drop(columns=["Autism"])
    y = df["Autism"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
X_train_pca, X_test_pca, y_train_pca, y_test_pca = prepare_data(pca_df)
X_train_pcoa, X_test_pcoa, y_train_pcoa, y_test_pcoa = prepare_data(pcoa_df)

# %%
# 📌 Entraîner un Random Forest sur PCA
rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train_pca)

# 📌 Entraîner un Random Forest sur PCoA
rf_pcoa = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pcoa.fit(X_train_pcoa, y_train_pcoa)

# 📌 Prédictions et performance
y_pred_pca = rf_pca.predict(X_test_pca)
y_pred_pcoa = rf_pcoa.predict(X_test_pcoa)

print("Performance sur données PCA :")
print(classification_report(y_test_pca, y_pred_pca))
print(f"Accuracy : {accuracy_score(y_test_pca, y_pred_pca):.2f}")

print("\nPerformance sur données PCoA :")
print(classification_report(y_test_pcoa, y_pred_pcoa))
print(f"Accuracy : {accuracy_score(y_test_pcoa, y_pred_pcoa):.2f}")

# %%
# 📌 Importance des features (PC1, PC2 ou PCoA1, PCoA2)
feature_importance_pca = pd.Series(rf_pca.feature_importances_, index=X_train_pca.columns)
feature_importance_pcoa = pd.Series(rf_pcoa.feature_importances_, index=X_train_pcoa.columns)

# 📌 Visualisation de l'importance des features
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x=feature_importance_pca.values, y=feature_importance_pca.index, ax=axes[0])
axes[0].set_title("Importance des features - PCA")
axes[0].set_xlabel("Score d'importance")

sns.barplot(x=feature_importance_pcoa.values, y=feature_importance_pcoa.index, ax=axes[1])
axes[1].set_title("Importance des features - PCoA")
axes[1].set_xlabel("Score d'importance")

plt.tight_layout()
plt.show()

# %% [markdown]
# # Optimization of RF on PCA dims

# %%
from tqdm import tqdm
# 📌 Tester différentes valeurs de n_components pour PCA
n_estimators_list = [50, 100, 200]  # Test de différents nombres d'arbres dans Random Forest
n_components_list = range(2, 50)
results = []

for n_estimators in tqdm(n_estimators_list):
    print(f"Testing n_estimators={n_estimators}")
    scores_per_n_components = []  # Liste pour stocker les scores pour chaque n_components

    for n in n_components_list:
        # Réduire les dimensions avec PCA
        pca = PCA(n_components=n)
        X_pca_reduced = pca.fit_transform(clr_transformed_df)  # Utiliser les données CLR transformées

        # Diviser les données en X (features) et y (target)
        X = X_pca_reduced
        y = metadata_df['Autism']

        # Initialiser Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        # Validation croisée pour évaluer la performance du modèle
        scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
        scores_per_n_components.append(scores.mean())
        
    results.append(scores_per_n_components)

        #print(f"n_components={n} => Accuracy (CV) : {scores.mean():.4f} +/- {scores.std():.4f}")


# %%
# 📌 Visualiser les scores en fonction de n_components
plt.figure(figsize=(10,4))
plt.plot(n_components_list, results[0], marker='o', label='n_estimators : 50')
#plt.plot(n_components_list, results[1], marker='o', label='n_estimators : 100')
plt.plot(n_components_list, results[2], marker='o', label='n_estimators : 200')
plt.title('Impact de n_components sur la performance de PCA + Random Forest')
plt.xlabel('n_components')
plt.ylabel('Accuracy (Validation croisée)')
plt.legend()
plt.ylim(0.7, 0.9)
plt.grid(True)
plt.show()

# %%
best_n_components = n_components_list[np.argmax(results[0])]
best_n_components

# %% [markdown]
# ## Eigen value of msp 

# %%
# 📌 Appliquer PCA (si ce n'est pas déjà fait)
pca = PCA(n_components=best_n_components)  # Nombre de composantes PCA à extraire
X_pca_reduced = pca.fit_transform(clr_transformed_df)  # Assure-toi d'utiliser les données CLR transformées

# %%
# 📌 Extraire les coefficients associés à la première composante principale (PC1)
pc1_contributions = pca.components_[0]  # Premier vecteur propre

# %%
# 📌 Créer un DataFrame pour afficher les contributions des espèces
species_contributions = pd.DataFrame({
    'msp': clr_transformed_df.columns,  # Noms des espèces (colonnes)
    'PC1_Contribution': pc1_contributions
})

# 📌 Trier les espèces en fonction de leur contribution à la PC1
species_contributions_sorted = species_contributions.sort_values(by='PC1_Contribution', ascending=False)

# %% [markdown]
# ## Importance des 10 top msp pour RF

# %%
# 📌 Sélectionner les 10 premières espèces ayant les plus fortes contributions
top_10_species = species_contributions_sorted.head(10)

# 📌 Récupérer la métadonnée (exemple : groupe cible de classification)
# Suppose que tu as une colonne 'group' qui représente les groupes (à ajuster selon ta structure de données)
top_10_species_data = clr_transformed_df[top_10_species['msp']]
y = metadata_df['Autism']  # Remplacer 'group' par la colonne de ta métadonnée cible
X = top_10_species_data  # Les 10 espèces comme caractéristiques d'entrée

# 📌 Créer et entraîner un modèle Random Forest (classification)
rf = RandomForestClassifier(n_estimators=50, random_state=42)

# 📌 Cross-validation pour évaluer la performance du modèle
cv_scores = cross_val_score(rf, X, y, cv=5)  # 5-fold cross-validation
print(f"Scores de validation croisée : {cv_scores}")
print(f"Accuracy moyenne : {np.mean(cv_scores)}")

# 📌 Afficher l'importance des caractéristiques pour le Random Forest
rf.fit(X, y)  # Entraînement final sur toutes les données
importances = rf.feature_importances_

top_10_species['importances'] = importances

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x='PC1_Contribution', y='msp', data=top_10_species, palette='viridis', ax=axes[0])
axes[0].set_title("Top 10 msp")
axes[0].set_xlabel("msp")


sns.barplot(x='importances', y="msp", data=top_10_species, color='skyblue',ax=axes[1])
axes[1].set_title("Importance des espèces les plus contributrices à la PC1 selon Random Forest")
axes[1].set_xlabel("Importance des caractéristiques")


plt.tight_layout()
plt.show()




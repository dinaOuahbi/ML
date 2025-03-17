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
pd_rel_abundance = pd_abundance_T / abs_abundance 

pd_rel_abundance.head()


# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(pd_rel_abundance)

# %%
test_train_split_SEED = 1970
X_train, X_test, y_train, y_test = train_test_split(pd_rel_abundance, binary_target, test_size=0.05, random_state=test_train_split_SEED,shuffle = True)

print(
    X_train.shape,
    X_test.shape,
)

# %%
k_values = list(range(1, 51))
cross_val_scores_bc = []

for k in k_values:
    knn = KNN_BrayCurtis(k=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cross_val_scores_bc.append(scores.mean())

optimal_k_bc = k_values[cross_val_scores_bc.index(max(cross_val_scores_bc))]

# %%
k_values = list(range(1, 51))
cross_val_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cross_val_scores.append(scores.mean())

optimal_k = k_values[cross_val_scores.index(max(cross_val_scores))]

# Plot the Cross-Validation Results
plt.figure(figsize=(10, 6))
plt.plot(k_values, cross_val_scores, marker='o', linestyle='-', color='b')
plt.plot(k_values, cross_val_scores_bc, marker='o', linestyle='-', color='r')
plt.xlabel('Value of k for k-NN')
plt.ylabel('Cross-Validation Accuracy')
plt.title('k-NN Cross-Validation Accuracy vs. k')
plt.axvline(x=optimal_k, color='b', linestyle='--', label=f'Optimal k using euclidean = {optimal_k}')
plt.axvline(x=optimal_k_bc, color='r', linestyle='--', label=f'Optimal k using BrayCurtis = {optimal_k_bc}')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Initialize the k-NN Classifier with the defined hyperparameters
knn_optimal = KNN_BrayCurtis(k=optimal_k_bc)

# Evaluate the classifier's accuracy on the test data
knn_optimal.fit(X_train, y_train)
knn_optimal_accuracy = knn_optimal.score(X_test, y_test)
print(f"Accuracy of k-NN Classifier: {knn_optimal_accuracy:.4f}")

# %%
# Créer une grille de points pour visualiser la frontière de décision
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Prédire les classes pour chaque point de la grille
knn_optimal.fit(X_pca, binary_target)
Z = knn_optimal.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Affichage
plt.figure(figsize=(8, 6))

# Tracer la frontière de décision
plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

# Tracer les points de données
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=binary_target, cmap="viridis", edgecolor='k', alpha=0.8)

# Labels et titre
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Frontière de décision KNN après PCA")
plt.colorbar(scatter, label="Classes")

plt.show()



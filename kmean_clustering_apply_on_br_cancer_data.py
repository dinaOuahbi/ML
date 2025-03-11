# %%
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("data/income.csv")
df.head()

# 

plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

#

scaler = MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
df.head()

#

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df.drop('Name', axis=1))
df['cluster'] = y_predicted

#
print(km.cluster_centers_)

#

sns.scatterplot(data=df, x="Age", y="Income($)", hue="cluster",palette="deep")
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
plt.legend()
plt.show()

#

sse = []
k_rng = range(1,df.shape[0]) #clusters
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit_predict(df.drop('Name', axis=1))
    sse.append(km.inertia_) # sum of squared error

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()



# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import cm
import seaborn as sns; sns.set()
import scipy

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics.cluster import adjusted_rand_score

# %% [markdown]
# data taken from https://www.kaggle.com/brunogrisci/breast-cancer-gene-expression-cumida
# 
# gene expression from breast cancer
# 
# curated from 30.000 studies from the Gene Expression Omnibus (GEO),
# 
# Dataset GSE45827 on breast cancer gene expression from CuMiDa
#         6 classes
#         54676 genes
#         151 samples
# 
# 
# The database make available various download options to be employed by other programs, as well for PCA and t-SNE results

# %%
def plot_tsne(xi, yi, data_structure='origine'):
    plt.figure(figsize=(12, 6))  # Taille plus raisonnable
    sns.scatterplot(
        x=xi, y=yi,
        hue=y,
        palette=sns.color_palette("tab10"),  # Palette plus claire
        legend="full",
        alpha=0.7,  # Transparence pour voir le chevauchement
        s=80,  # Taille des points plus grande
        edgecolor='black',  # Bordure blanche pour mieux distinguer les points
        linewidth=0.5,
        ) # Épaisseur de la bordure

        # Ajout de labels et d'un titre
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.title(f"t-SNE Visualization of {data_structure} Data", fontsize=16, fontweight='bold')

    # Déplacer la légende en dehors du plot
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Améliorer le style général
    sns.despine()  # Supprime les bordures inutiles
    plt.grid(True, linestyle='--', alpha=0.5)  # Ajouter une grille discrète

    # Sauvegarde du plot
    plt.tight_layout()  # Ajuste automatiquement les marges
    plt.savefig(f'issues/t-SNE_breast_GEO_{data_structure}.png', dpi=300, bbox_inches="tight")  # Qualité HD et légende bien placée
    plt.show()

def plot_silhouette(silhouette_vals,results, labels, tech='t-SNE'):
    # Initialisation
    n_clusters = len(np.unique(labels))
    yticks = []
    y_ax_lower = 0

    # Tracer des barres horizontales pour chaque cluster
    for i, c in enumerate(np.unique(labels)):
        c_silhouette_vals = silhouette_vals[labels == c]
        c_silhouette_vals.sort()
        y_ax_upper = y_ax_lower + len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower = y_ax_upper

    # Moyenne des scores de silhouette
    silhouette_avg = np.mean(silhouette_vals)

    silhouette_score = metrics.silhouette_score(results, labels, metric='euclidean')

    # Affichage de la ligne moyenne
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    # Personnalisation de l'axe Y et des labels
    plt.yticks(yticks, np.unique(labels) + 1)
    plt.ylabel(f'{tech} Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.title(f"silhouette_score : {silhouette_score}", fontsize=10)

    # Sauvegarde du plot
    plt.tight_layout()
    plt.savefig(f'issues/silhouette_plot_{tech}.png')
    plt.show()

# %%
df = pd.read_csv('data/Breast_from_GEO/Breast_GSE45827.csv')
df.head()

# %%
df['samples'].unique().shape

# %%
# Création du pie chart avec labels et pourcentages
df['type'].value_counts().plot.pie(
    autopct='%1.1f%%',  # Afficher les pourcentages
    startangle=90,      # Angle de départ pour éviter une coupure
    cmap='Set3',        # Palette de couleurs
    figsize=(6, 6)      # Ajuster la taille de la figure
)

plt.ylabel('')  # Supprimer le label de l'axe Y
plt.title("Distribution of BR cancer labels")
plt.savefig(f'issues/br_cancer_labels.png', dpi=300)
plt.show()

# %%
y = df['type']
data = df.select_dtypes('float').values

#scale our data such that each feature has unit variance. This is necessary because fitting algorithms highly depend on the scaling of the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#perform a calculation and plotting of the cluster errors to see whether 6 is really the optimal size for k.

# %%
from tqdm import tqdm
cluster_range = range(1, 16)
cluster_errors = []

for num_clusters in tqdm(cluster_range):
    clusters = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    clusters.fit(scaled_data)

    #The total sum of squared distances of every data point from respective centroid is also called inertia
    cluster_errors.append(clusters.inertia_)


clusters_df = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": cluster_errors})

print(clusters_df)

#Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(clusters_df["num_clusters"], clusters_df["cluster_errors"], marker='o', linestyle='-')
plt.xlabel("Nombre de clusters")
plt.ylabel("Erreur d'inertie")
plt.title("Méthode du coude pour choisir k")
plt.show()

# %%
km = KMeans(n_clusters=6, random_state=0)
labels = km.fit_predict(scaled_data)
centroids = km.cluster_centers_
df_labels = pd.DataFrame(km.labels_ , columns = list(['label']))
df_labels['label'] = df_labels['label'].astype('category')
df_labeled = df.join(df_labels)
df_labeled.head()

# %% [markdown]
# Cette fonction ***find_permutation*** sert à réaligner les labels prédits par K-Means avec les vrais labels en trouvant la meilleure correspondance entre eux.
# 
# **Pourquoi est-ce nécessaire ?**
# L'algorithme K-Means attribue des labels arbitraires aux clusters (par exemple, un cluster peut être "0" alors qu'il correspond en réalité à la classe "chien").
# Comme K-Means ne connaît pas les vraies étiquettes, il faut trouver la correspondance entre les clusters et les vraies classes avant d’évaluer les performances.

# %%
#Since the k-means algorithm doesn´t have any knowledge on the true cluster labels, the permutations need to be found before comparing to the true labels.
def find_permutation(n_clusters, real_labels, labels):
    permutation = {}
    assigned_labels = set()

    # Trouver les étiquettes dominantes pour chaque cluster
    for i in range(n_clusters):
        idx = labels == i
        unique_labels, counts = np.unique(real_labels[idx], return_counts=True)

        # Trier les labels par fréquence décroissante
        sorted_indices = np.argsort(-counts)
        for idx in sorted_indices:
            new_label = unique_labels[idx]
            if new_label not in assigned_labels:
                permutation[i] = new_label
                assigned_labels.add(new_label)
                break  # Passer au cluster suivant

    # Si certains clusters n'ont pas encore de label, leur attribuer un label restant
    all_labels = set(np.unique(real_labels))  # Tous les labels possibles
    remaining_labels = list(all_labels - assigned_labels)

    for i in range(n_clusters):
        if i not in permutation:
            if remaining_labels:  # S'il reste des labels non assignés
                permutation[i] = remaining_labels.pop(0)
            else:  # Si tout est déjà utilisé, mettre un label par défaut
                permutation[i] = "Unknown"

    return [permutation.get(i, "Unknown") for i in range(n_clusters)]


# %%
permutation = find_permutation(6, y, km.labels_)
permutation

# %%
new_labels = [ permutation[label] for label in km.labels_]   # permute the labels
print("Accuracy score is", accuracy_score(y, new_labels))

# %%
# plot confusion matrix
mat = confusion_matrix(y, new_labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=permutation,
            yticklabels=permutation)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.savefig('issues/confustion_matrix_original_data.png')

# %%
silhouette_vals = silhouette_samples(data,
                                      labels,
                                      metric='euclidean')

plot_silhouette(silhouette_vals,data, labels, tech='originalData')

# %%
kmeansSilhouette_Score = metrics.silhouette_score(data, labels, metric='euclidean')
kmeansSilhouette_Score

# %%
## Apply PCA before clustering
pca_plot = PCA().fit(scaled_data)
plt.plot(np.cumsum(pca_plot.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# %% [markdown]
# PCA didn´t seem to be a good approach as it would need 100 components to explain most of the data

# %% [markdown]
# ## Apply t-Sne

# %%
# original values
tsne = TSNE(random_state=0)
tsne_result = tsne.fit_transform(data)
xi = tsne_result[:, 0]
yi = tsne_result[:, 1]
plot_tsne(xi, yi, data_structure='origine')


# %%
# original values
tsne = TSNE(random_state=0)
tsne_result_scaled = tsne.fit_transform(scaled_data)
xi_scaled = tsne_result_scaled[:, 0]
yi_scaled = tsne_result_scaled[:, 1]
plot_tsne(xi_scaled, yi_scaled, data_structure='scaled')

# %% [markdown]
# t-SNE does better on original data than on scaled data.

# %%
# kmean on tsne result
km_tsne = KMeans(n_clusters = 6, random_state=0)
labels_tsne = km_tsne.fit_predict(tsne_result)
df_labels_tsne = pd.DataFrame(km_tsne.labels_ , columns = list(['label']))
df_labels_tsne['label'] = df_labels_tsne['label'].astype('category')
df_labels_tsne.head()

# %%
df_labels_tsne['label'].value_counts()

# %%
silhouette_vals = silhouette_samples(tsne_result, labels_tsne, metric='euclidean')
plot_silhouette(silhouette_vals,tsne_result, labels_tsne, tech='t-SNE')


# %%
kmeansSilhouette_Score = metrics.silhouette_score(tsne_result, labels_tsne, metric='euclidean')
kmeansSilhouette_Score



# %%

# %% [markdown]
# Dans ce projet, nous cherchons à implémenter un modèle de régression linéaire et ses extensions sur un ensemble de données sur les voitures, afin d'améliorer au maximum les performances du modèle.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from scipy.special import inv_boxcox

from assumption_utiles import *

# %%
df = pd.read_csv('data/Car_data/car data.csv')
df.head()

# %%
# indep var = Selling_Price
print(df.shape)
df.describe(include='number')

# %%
df.describe(include='object')

# %%
df.drop('Car_Name', axis=1, inplace=True)

# %%
df.insert(0, "Age", df["Year"].max()+1-df["Year"] )
df.drop('Year', axis=1, inplace=True)
df.head()

# %%
# outliers detect
sns.set_style('darkgrid')
colors = ['#0055ff', '#ff7000', '#23bf00']
CustomPalette = sns.set_palette(sns.color_palette(colors))

OrderedCols = np.concatenate([df.select_dtypes(exclude='object').columns.values, 
                              df.select_dtypes(include='object').columns.values])
OrderedCols

# %%
fig, ax = plt.subplots(2, 4, figsize=(15,7),dpi=100)

for i,col in enumerate(OrderedCols):
    x = i//4
    y = i%4
    if i<5:
        sns.boxplot(data=df, y=col, ax=ax[x,y])
        ax[x,y].yaxis.label.set_size(15)
    else:
        sns.boxplot(data=df, x=col, y='Selling_Price', ax=ax[x,y])
        ax[x,y].xaxis.label.set_size(15)
        ax[x,y].yaxis.label.set_size(15)

plt.tight_layout()    
plt.show()

# %% [markdown]
# IQR=Q3−Q1
# Selon la règle de l'IQR, les valeurs sont considérées comme des outliers si elles sont situées en dehors des bornes suivantes :
# 
# Q1 - 1.5 * IQR     : borne inf
# 
# Q1 + 1.5 * IQR       borne sup

# %%
outliers_indexes = []
target = 'Selling_Price'

for col in df.select_dtypes(include='object').columns:
    for cat in df[col].unique():
        df1 = df[df[col] == cat]
        q1 = df1[target].quantile(0.25)
        q3 = df1[target].quantile(0.75)
        iqr = q3-q1
        maximum = q3 + (1.5 * iqr)
        minimum = q1 - (1.5 * iqr)
        outlier_samples = df1[(df1[target] < minimum) | (df1[target] > maximum)]
        outliers_indexes.extend(outlier_samples.index.tolist())

print(outliers_indexes)

# %%
for col in df.select_dtypes(exclude='object').columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    maximum = q3 + (1.5 * iqr)
    minimum = q1 - (1.5 * iqr)
    outlier_samples = df[(df[col] < minimum) | (df[col] > maximum)]
    outliers_indexes.extend(outlier_samples.index.tolist())
    
outliers_indexes = list(set(outliers_indexes))
print('{} outliers were identified, whose indices are:\n\n{}'.format(len(outliers_indexes), outliers_indexes))


# %%
# it’s important to investigate the nature of the outlier before deciding whether to drop it or not.
# Outliers Labeling
df1 = df.copy()
df1['label'] = 'Normal'
df1.loc[outliers_indexes,'label'] = 'Outlier'

# Removing Outliers
removing_indexes = []
removing_indexes.extend(df1[df1[target]>33].index)
removing_indexes.extend(df1[df1['Kms_Driven']>400000].index)
df1.loc[removing_indexes,'label'] = 'Removing'
df1.head()

# %%
# Plot
target = 'Selling_Price'
features = df.columns.drop(target)
colors = ['#0055ff','#ff7000','#23bf00']
CustomPalette = sns.set_palette(sns.color_palette(colors))
fig, ax = plt.subplots(nrows=3 ,ncols=3, figsize=(15,12), dpi=200)

for i in range(len(features)):
    x=i//3 # 3 columns
    y=i%3 #after tree ==> new row
    sns.scatterplot(data=df1, x=features[i], y=target, hue='label', ax=ax[x,y])
    ax[x,y].set_title('{} vs. {}'.format(target, features[i]), size = 15)
    ax[x,y].set_xlabel(features[i], size = 12)
    ax[x,y].set_ylabel(target, size = 12)
    ax[x,y].grid()

ax[2, 1].axis('off')
ax[2, 2].axis('off')
plt.tight_layout()
plt.show()

# %%
# Since Linear Regression is sensitive to outliers, we will drop them.
removing_indexes = list(set(removing_indexes))
removing_indexes

# %%
# NAN
print(df.isnull().sum())

# %%
# drop outliers
df1 = df.copy()
df1.drop(removing_indexes, inplace=True)
df1.reset_index(drop=True, inplace=True)

# %%
# Ecoding
CatCols = ['Fuel_Type', 'Seller_Type', 'Transmission']

df1 = pd.get_dummies(df1, columns=CatCols, drop_first=True)
df1.head(5)



# %%
# corr analysis
target = 'Selling_Price'
cmap = sns.diverging_palette(125, 28, s=100, l=65, sep=50, as_cmap=True)
fig, ax = plt.subplots(figsize=(9, 8), dpi=80)
ax = sns.heatmap(pd.concat([df1.drop(target,axis=1), df1[target]],axis=1).corr(), annot=True, cmap=cmap)
plt.show()

# %% [markdown]
# Target variable "Selling Price" is highly correlated with Present_Price
# 
# Some independent variables like Fuel_Type_Petrol and Fuel_Type_Disel are highly correlated, which is called Multicollinearity.

# %%
# Linear regression
X = df1.drop('Selling_Price', axis=1)
y = df1['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

y_test_actual = y_test

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



# %% [markdown]
# La normalisation des données doit être effectuée uniquement à partir des données d'entraînement, et les mêmes paramètres de transformation doivent ensuite être appliqués aux données de test. Cela garantit que le test reste indépendant et fiable pour évaluer la performance réelle du modèle.

# %%
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

pd.DataFrame(data = np.append(linear_reg.intercept_ , linear_reg.coef_), 
             index = ['Intercept']+[col+" Coef." for col in X.columns], columns=['Value']).sort_values('Value', ascending=False)


# %%
res1 = pd.concat([model_evaluation(linear_reg, X_train_scaled, y_train, 'Linear Reg. Train'),
model_evaluation(linear_reg, X_test_scaled, y_test, 'Linear Reg. Test')], axis=1).reset_index()

df_melted = res1.melt(id_vars=['index'], var_name='Dataset', value_name='Valeur')


# %%
df_melted

# %%
# Création du barplot
plt.figure(figsize=(6, 4))
sns.barplot(data=df_melted, x='Dataset', y='Valeur', hue='index', palette='Set2')


# %% [markdown]
# Le score R² de 88,72 % indique que 88 % de la variance des données peut être expliquée par le modèle
# 
# Le modèle est capable de bien prédire les valeurs de la variable cible.

# %%
linear_reg_cv = LinearRegression()
scaler = StandardScaler()
pipeline = make_pipeline(StandardScaler(),  LinearRegression())

"""
Un pipeline est une manière efficace d'éviter la fuite de données.
Il garantit que les méthodes de transformation ou de modélisation sont appliquées uniquement sur les données appropriées.
"""

kf = KFold(n_splits=6, shuffle=True, random_state=0) 
scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
result = cross_validate(pipeline, X, y, cv=kf, return_train_score=True, scoring=scoring)

MAE_mean = (-result['test_neg_mean_absolute_error']).mean()
MAE_std = (-result['test_neg_mean_absolute_error']).std()
MSE_mean = (-result['test_neg_mean_squared_error']).mean()
MSE_std = (-result['test_neg_mean_squared_error']).std()
RMSE_mean = (-result['test_neg_root_mean_squared_error']).mean()
RMSE_std = (-result['test_neg_root_mean_squared_error']).std()
R2_Score_mean = result['test_r2'].mean()
R2_Score_std = result['test_r2'].std()

pd.DataFrame({'Mean': [MAE_mean,MSE_mean,RMSE_mean,R2_Score_mean], 'Std': [MAE_std,MSE_std,RMSE_std,R2_Score_std]},
             index=['MAE', 'MSE', 'RMSE' ,'R2-Score'])

# %% [markdown]
# # Assumptions 
# 
# Il est essentiel de vérifier les hypothèses de la régression linéaire, car si elles ne sont pas respectées, les résultats peuvent être faussés et leur interprétation incorrecte.
# 
# Les principales hypothèses à vérifier sont :
# 
#         ✅ Linéarité : La relation entre les variables doit être linéaire.
# 
#         ✅ Normalité des résidus : Les erreurs doivent suivre une distribution normale.
# 
#         ✅ Homoscedasticité : La variance des résidus doit être constante.
# 
#         ✅ Absence d’autocorrélation : Les erreurs ne doivent pas être corrélées entre elles.
# 
#         ✅ Faible multicolinéarité : Les variables indépendantes ne doivent pas être trop corrélées entre elles.
# 
# Si ces conditions ne sont pas remplies, des corrections sont nécessaires (transformation des variables, régularisation, modèles alternatifs).

# %% [markdown]
# ## Assumption 1 : linearity
#  Si la relation entre les variables indépendantes et la variable cible n'est pas linéaire, le modèle fera de grosses erreurs de prédiction (sous-ajustement ou underfitting).
# 
# 🔹 Comment détecter une non-linéarité ?
# 
# ✅ Graphique des valeurs réelles vs. prédictions → Les points doivent être alignés symétriquement autour de la diagonale.
# 
# ✅ Graphique des résidus vs. prédictions → Les résidus doivent être répartis aléatoirement autour d’une ligne horizontale.
# 
# ✅ Vérification de la variance → Elle doit rester constante sur l’ensemble des valeurs prédites.
# 
# Si ces conditions ne sont pas respectées, il faut envisager :
# 
# Une transformation des variables (ex. logarithme, Box-Cox).
# Une régression non linéaire (ex. polynomiale, arbre de décision).

# %%
linear_assumption(linear_reg, X_test_scaled, y_test)

# %% [markdown]
# The inspection of the plots shows that the linearity assumption is not satisfied.
# 
# Potential solutions:
# 
# Applying nonlinear transformations
# Adding polynomial terms to some of the predictors

# %% [markdown]
# ## Assumption 2 : Normality of residuals
# 
# Les erreurs du modèle doivent suivre une distribution normale avec une moyenne de zéro. Si ce n'est pas le cas, cela peut être dû à des variables très éloignées de la normalité, à une violation de l'hypothèse de linéarité ou à la présence de valeurs extrêmes.
# 
# Si cette hypothèse est violée, les intervalles de confiance peuvent être trop larges ou trop étroits. Toutefois, si l'objectif est uniquement d'estimer les coefficients et de faire des prédictions (en minimisant l'erreur quadratique moyenne), cette hypothèse n'est pas indispensable. Mais pour faire des inférences valides ou évaluer la probabilité d'une erreur dans une direction précise, elle doit être respectée.
# 
# Pour vérifier cette hypothèse, on peut :
# 
# Observer l'histogramme des résidus.
# Utiliser un Q-Q plot (Quantile-Quantile plot) pour comparer les résidus à une distribution normale.
# Effectuer le test d'Anderson-Darling, qui mesure à quel point les résidus s'écartent d'une distribution normale.

# %%

normal_errors_assumption(linear_reg, X_test_scaled, y_test)


# %% [markdown]
# Dans un QQ plot des résidus :
# 
# Une forme en arc ("bow-shaped") indique une asymétrie excessive des résidus.
# Une forme en "S" suggère une kurtosis excessive, c'est-à-dire trop ou trop peu de valeurs extrêmes dans les deux directions.
# 
# Une moyenne des résidus non nulle, une asymétrie positive et une forme en "S" dans le QQ plot indiquent que les résidus ne suivent pas une distribution normale.
# 
# solutions : 
# - non linear transform of target and features
# - remove outliers

# %% [markdown]
# ## Assumption 3 - No Perfect Multicollinearity
# 
# La multicolinéarité se produit lorsque les variables indépendantes sont corrélées entre elles. Cela complique l'estimation de leur relation avec la variable dépendante, car elles varient ensemble.
# 
# Les coefficients deviennent instables et très sensibles aux changements dans le modèle, ce qui réduit leur précision et augmente leur erreur standard. Cela peut les rendre statistiquement insignifiants alors qu'ils sont en réalité significatifs. De plus, ces variations simultanées peuvent provoquer un surajustement du modèle et nuire à sa performance.
# 
# - we can use Heatmap or VIF 
# 
# ***L'Interpretation du VIF (Variance Inflation Factor) :***
# 
# La racine carrée du VIF d'une variable indique de combien l'erreur standard est amplifiée à cause de sa corrélation avec les autres variables du modèle.
# 
# Plus le VIF est élevé, plus la variable est corrélée aux autres.
# Une règle générale : si VIF > 10, la multicolinéarité est forte et peut poser problème.

# %%
multicollinearity_assumption(X.astype('float'))

# %% [markdown]
# solutions:
# 
#     utiliser la regularisation
#     enlever les feature avec vif elevées
#     Using PCA 

# %% [markdown]
# ## Assumption 4 - No Autocorrelation of Residuals
# 
# L'autocorrélation est la corrélation d'une variable avec elle-même à différents moments ou positions.
# 
# En statistique, elle mesure si les valeurs successives d'une série de données sont liées. Par exemple, dans une série temporelle, une forte autocorrélation signifie que la valeur actuelle dépend fortement des valeurs passées.
# 
# Dans un modèle, l'autocorrélation des résidus indique que les erreurs ne sont pas indépendantes, ce qui peut biaiser les résultats et réduire la fiabilité des prédictions.
# Une cause possible est la violation de l'hypothèse de linéarité.
# 
# **- Durbin-Watson test**
# 
#     Values of 1.5 < d < 2.5 means that there is no autocorrelation in the data
# 

# %%
autocorrelation_assumption(linear_reg, X_test_scaled, y_test)

# %% [markdown]
# no autocorrelation
# 
# otherwise : if there is autoCorr, add interaction terms

# %% [markdown]
# ## Assumption 5 - Homoscedasticity
# 
# L'homoscedasticité signifie que l'écart (ou la "dispertion") des erreurs du modèle est constant, peu importe la valeur de la variable cible (ce que tu essaies de prédire).
# 
# Si les erreurs (résidus) n'ont pas une variance constante, on parle de hétéroscédasticité. Cela rend les intervalles de confiance (les plages dans lesquelles on pense que les vraies valeurs se trouvent) parfois trop larges ou trop étroits, ce qui rend les prévisions moins fiables. Cela peut aussi rendre certaines parties des données plus influentes qu'elles ne devraient l'être.
# 
# Pour vérifier si tes erreurs sont homoscédastiques (variance constante), tu peux faire un graphique des résidus comparés aux valeurs prédites. Si les points sont dispersés au hasard sans forme particulière (pas d'augmentation ou de diminution dans les erreurs), cela montre que tes données respectent l'homoscedasticité.
# 

# %%
homoscedasticity_assumption(linear_reg, X_test_scaled, y_test)

# %% [markdown]
#  la variance des résidus n'est pas uniforme, car la ligne orange (tendance ou moyenne des residus) n'est pas plate. Cela montre que l'hypothèse d'homoscedasticité n'est pas respectée, ce qui peut entraîner des problèmes dans l'estimation des erreurs et des intervalles de confiance.
# 
# **Solutions***
# - outliers rm
# - log transform
# - polynomial reg
# 
# 
# **2nd-order Polynomial Regression:**
# C'est une régression polynomiale où on ajoute des termes quadratiques (de degré 2) des variables indépendantes dans le modèle.
# Overfiting si les termes deviennent plus complexes
# 
# 
# **Ridge Regression :**
# une régression linéaire avec une régularisation L2. La régularisation L2 ajoute une pénalité basée sur le carré des coefficients dans la fonction de coût.
# Utiles en presence de bcp de correlations fortes entre variables indep
# 
# **Lasso Regression :**
# une régression linéaire avec une régularisation L1. La régularisation L1 ajoute une pénalité basée sur la valeur absolue des coefficients, ce qui peut forcer certains coefficients à zéro.
# Utiles quand tu veux reduire le nombre de variables dans le modele.

# %% [markdown]
# ## Preprocess
# 
# - rm Fuel_Type_Petrol  (multicollinearity assumption)
# - box-cox transformation on the entire dataset (normality assumptions)
# - polynomial regression (homoscedasticity and normality assumption)
# - regularization (avoid overfiting)

# %%
del df1['Fuel_Type_Petrol']
y_test_pred = linear_reg.predict(X_test_scaled)
df_comp = pd.DataFrame({'Actual':y_test_actual, 'Predicted':y_test_pred})
df_comp.head()

# %%
compare_plot(df_comp)

# %% [markdown]
# this show the prediction error of the model on the test data.
# 
# Also, out of 90 test samples, Selling_Price has been predicted negatively in 6 cases. A negative prediction for Selling_Price is disappointing.

# %%
# Box cox transform

# transform x_train et get lambda
fitted_lambda = pd.Series(np.zeros(len(df1.columns), dtype=np.float64), index=df1.columns)

y_train, fitted_lambda['Selling_Price'] = stats.boxcox(y_train+1)
for col in X_train.columns:
    X_train[col], fitted_lambda[col] = stats.boxcox(X_train[col]+1)
    
fitted_lambda

# %%
# transform x-test with lambda
y_test = stats.boxcox(y_test+1, fitted_lambda['Selling_Price'])
for col in X_test.columns:
    X_test[col] = stats.boxcox(X_test[col]+1, fitted_lambda[col])

# %%
y_train = pd.DataFrame(y_train, index=X_train.index, columns=['Selling_Price'])
y_test = pd.DataFrame(y_test, index=X_test.index, columns=['Selling_Price'])

X_boxcox = pd.concat([X_train, X_test])
y_boxcox = pd.concat([y_train, y_test])

df_boxcox = pd.concat([X_boxcox, y_boxcox], axis=1)
df_boxcox.sort_index(inplace=True)

del df_boxcox['Fuel_Type_Petrol']

# %%
fig, ax = plt.subplots(2, 4, figsize=(15,8), dpi=100)
columns = ['Selling_Price', 'Present_Price', 'Kms_Driven', 'Age']

for i,col in enumerate(columns):
    sns.kdeplot(df1[col], label="Non-Normal", fill=True, color='#0055ff', linewidth=2, ax=ax[0,i])
    sns.kdeplot(df_boxcox[col], label="Normal", fill=True, color='#23bf00', linewidth=2, ax=ax[1,i])  
    ax[0,i].set_xlabel('', fontsize=15)
    ax[1,i].set_xlabel(col, fontsize=15, fontweight='bold')
    ax[0,i].legend(loc="upper right")
    ax[1,i].legend(loc="upper right")

ax[0,2].tick_params(axis='x', labelrotation = 20)
plt.suptitle('Data Transformation using Box-Cox', fontsize=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Build 2nd-order Polynomial Regression
# 
# C'est une régression polynomiale où on ajoute des termes quadratiques (de degré 2) des variables indépendantes dans le modèle.
# Overfiting si les termes deviennent plus complexes
# 

# %%
X = df_boxcox.drop('Selling_Price', axis=1)
y = df_boxcox['Selling_Price']

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X_poly = pd.DataFrame(X_poly, columns=poly_features.get_feature_names_out(X.columns))

poly_features_names = poly_features.get_feature_names_out(X.columns)
print(len(poly_features_names))

X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
scaler.fit(X_poly_train)

X_poly_train = scaler.transform(X_poly_train)
X_poly_train = pd.DataFrame(X_poly_train, columns=poly_features_names)

X_poly_test = scaler.transform(X_poly_test)
X_poly_test = pd.DataFrame(X_poly_test, columns=poly_features_names)

# create poly regression using linear regression
polynomial_reg = LinearRegression()
polynomial_reg.fit(X_poly_train, y_poly_train)

model_evaluation(polynomial_reg, X_poly_test, y_poly_test, 'Polynomial Reg. Test')

# %%
model_evaluation(polynomial_reg, X_poly_train, y_poly_train, 'Polynomial Reg. Train')

# %%
res2 = pd.concat([
    model_evaluation(polynomial_reg, X_poly_train, y_poly_train, 'Polynomial Reg. Train'),
    model_evaluation(polynomial_reg, X_poly_test, y_poly_test, 'Polynomial Reg. Test')
    ], axis=1).reset_index()

df_melted2 = res2.melt(id_vars=['index'], var_name='Dataset', value_name='Valeur')

merge_res = pd.concat([res1, res2.drop('index', axis=1)], axis=1)
merge_melt = merge_res.melt(id_vars=['index'], var_name='Dataset', value_name='Valeur')

# %% [markdown]
# As can be seen, using boxcox transformation and production of second-order features has improved the model performance greatly!

# %%
linear_assumption(polynomial_reg, X_poly_test, y_poly_test)
normal_errors_assumption(polynomial_reg, X_poly_test, y_poly_test)
warnings.simplefilter(action='ignore')
multicollinearity_assumption(X_poly).T
autocorrelation_assumption(polynomial_reg, X_poly_test, y_poly_test)
homoscedasticity_assumption(polynomial_reg, X_poly_test, y_poly_test)

# %% [markdown]
# ## Ridge regression 
# 
# pour corriger la multicolinearité, on utilise la regularisation
# 
# une régression linéaire avec une régularisation L2. La régularisation L2 ajoute une pénalité basée sur le carré des coefficients dans la fonction de coût.
# Utiles en presence de bcp de correlations fortes entre variables indep

# %%
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


alphas = 10**np.linspace(10,-2,100)*0.5

ridge_cv_model = RidgeCV(alphas = alphas, cv = 3, scoring = 'neg_mean_squared_error')                        
ridge_cv_model.fit(X_train, y_train)

model_evaluation(ridge_cv_model, X_test, y_test, 'Ridge Reg. Test')

# %%
res3 = pd.concat([
    model_evaluation(ridge_cv_model, X_train, y_train, 'Ridge Reg. Train'),
    model_evaluation(ridge_cv_model, X_test, y_test, 'Ridge Reg. Test'),
    ], axis=1).reset_index()

merge_res = pd.concat([res1, res2.drop('index', axis=1),res3.drop('index', axis=1)], axis=1)
merge_melt = merge_res.melt(id_vars=['index'], var_name='Dataset', value_name='Valeur')

# %% [markdown]
# # Buit higher reg poly model
# 

# %%
scores, feature_num = poly_check(6, X, y)


# %%
# Plot1
fig, ax = plt.subplots(1, 2, figsize=(15,6), dpi=200, gridspec_kw={'width_ratios': [3, 1]})

sns.pointplot(x=scores['Degree'], y=scores['Ridge'], color='#ff7000', label='Ridge', ax=ax[0])
sns.pointplot(x=scores['Degree'], y=scores['Lasso'], color='#0055ff', label='Lasso', ax=ax[0])
sns.pointplot(x=scores['Degree'], y=scores['ElasticNet'], color='#23bf00', label='Elastic-Net', ax=ax[0])
ax[0].set_xlabel('Polynomial Degree', fontsize=12)
ax[0].set_ylabel('R2-Score', fontsize=12)
ax[0].legend(loc='upper left')
ax[0].grid(axis='x')
ax[0].set_ylim([0.96, 0.99])

# Annotate Points
for i,j,f in zip(scores['Degree']-2, scores['Ridge'], feature_num['Ridge']):
    ax[0].text(i, j+0.0008, str(f), ha='center', color='#ff7000', weight='bold', fontsize=15)

for i,j,f in zip(scores['Degree']-2, scores['Lasso'], feature_num['Lasso']):
    ax[0].text(i, j-0.0015, str(f), ha='center', color='#0055ff', weight='bold', fontsize=15)
    
for i,j,f in zip(scores['Degree']-2, scores['ElasticNet'], feature_num['ElasticNet']):
    ax[0].text(i, j+0.0008, str(f), ha='center', color='#23bf00', weight='bold', fontsize=15)
    
# Plot2    
table = ax[1].table(cellText=scores.values, colLabels=scores.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(13)
ax[1].set_xticks([])
ax[1].set_yticks([])
table.scale(1, 2)

plt.suptitle('R2-Score vs. Polynomial Degree on Test Data', fontsize=20)
plt.tight_layout()
plt.show()


# %%
# optimal model

poly_features = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly_features.fit_transform(X)
X_poly = pd.DataFrame(X_poly, columns=poly_features.get_feature_names_out(X.columns))

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

final_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, cv=4, max_iter=100000)

final_model.fit(X_train, y_train)
print(final_model.l1_ratio_,
     final_model.alpha_ )




# %%
res4 = pd.concat([
    model_evaluation(final_model, X_train, y_train, 'Final Model. Train'),
    model_evaluation(final_model, X_test, y_test, 'Final Model. Test'),
    ], axis=1).reset_index()

merge_res = pd.concat([res1, res2.drop('index', axis=1),res4.drop('index', axis=1)], axis=1)
merge_melt = merge_res.melt(id_vars=['index'], var_name='Dataset', value_name='Valeur')

# %%
plt.figure(figsize=(8, 4))
ax = sns.barplot(data=merge_melt, x='Dataset', y='Valeur', hue='index', palette='Set2')

# Adding the values on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.4f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                fontsize=8, color='black', 
                xytext=(0, 5), textcoords='offset points', rotation=45)

# Add dashed vertical lines between each x-label
x_ticks = range(len(merge_melt['Dataset'].unique()) - 1)  # Get the number of x-ticks
for tick in x_ticks:
    plt.axvline(x=tick + 0.5, color='Gray', linestyle='--', linewidth=0.5)
# Add labels and title
plt.xlabel('Dataset')
plt.ylabel('Value')
plt.title('Linear Regression Performance')
plt.yticks(np.arange(0, 5, 0.5))
plt.legend(title="Scores")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)



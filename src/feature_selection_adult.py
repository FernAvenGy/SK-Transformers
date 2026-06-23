import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr


# Cargar el dataset de Adult
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(BASE_DIR, "Data", "Adult")
adult_filename = os.path.join(data_folder, "adult.data")

adult = pd.read_csv(
    adult_filename,
    header=None,
    names=[
        "Age", "Work-Class", "fnlwgt",
        "Education", "Education-Num",
        "Marital-Status", "Occupation",
        "Relationship", "Race", "Sex",
        "Capital-gain", "Capital-loss",
        "Hours-per-week", "Native-Country",
        "Earnings-Raw"
    ]
)

print("========= Preprocesamiento de la BD ========= ")
# Eliminar filas vacías
adult.dropna(how='all', inplace=True)

# Ver nombres de columnas
print("Nombre de columnas:\n", adult.columns)

# Estadísticas de la columna Hours-per-week
print("\nEstadísticas de la columna Hours-per-week\n")
print(adult["Hours-per-week"].describe())

# Mediana de Education-Num
print("\nMediana de Education-Num: ", adult["Education-Num"].median())

# Ver valores únicos de Work-Class
print("\nValores únicos de Work-Class: ", adult["Work-Class"].unique())

# Crear feature categórica a partir de una numérica (discretización)
adult["LongHours"] = adult["Hours-per-week"] > 40

print("\nBase de datos modificada")
print(adult)


# ── Selección de características — VarianceThreshold ──────────
print("\n========= Prueba de VarianceThreshold ========= ")

X = np.arange(30).reshape((10, 3))
print("\nMatriz original")
print(X)

# Fijar toda la segunda columna al valor 1 (sin varianza)
X[:, 1] = 1
print("\nMatriz modificada")
print(X)

vt = VarianceThreshold()
Xt = vt.fit_transform(X)

# Ver varianzas de cada columna
print("\nVarianza de columnas; ")
print(vt.variances_, "\n")


# ── Selección de las mejores features individuales ──────────
X = adult[["Age", "Education-Num", "Capital-gain",
           "Capital-loss", "Hours-per-week"]].values

y = (adult["Earnings-Raw"] == ' >50K').values

# Selección con chi2
transformer_chi2 = SelectKBest(score_func=chi2, k=3)
Xt_chi2 = transformer_chi2.fit_transform(X, y)
print("========= Scores de chi2 ========= ")
print(transformer_chi2.scores_, "\n")

# Selección con correlación de Pearson
def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = pearsonr(X[:, column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))

transformer_pearson = SelectKBest(score_func=multivariate_pearsonr, k=3)
Xt_pearson = transformer_pearson.fit_transform(X, y)
print("========= Scores de pearson ========= ")
print(transformer_pearson.scores_, "\n")

# Comparar rendimiento entre chi2 y Pearson
clf = DecisionTreeClassifier(random_state=14)
scores_pearson = cross_val_score(clf, Xt_pearson, y, scoring='accuracy')
scores_chi2 = cross_val_score(clf, Xt_chi2, y, scoring='accuracy')

print("========= Comparación de accuracy de clasificador ========= ")
print("Accuracy chi2: ", scores_chi2.mean())
print("Accuracy pearson: ", scores_pearson.mean())

feature_cols = ["Age", "Education-Num", "Capital-gain", "Capital-loss", "Hours-per-week"]
selected = [col for col, s in zip(feature_cols, transformer_chi2.get_support()) if s]
print("\nFeatures seleccionadas por chi2:", selected)


# ── Predicción en nuevos registros - clf con features de chi2 ──────────
# Entrenar el clf con las features de chi2 (83% acc)
clf.fit(Xt_chi2, y) 

# Nuevo registro ficticio 
print("\n========= Predicción de salario en nuevo registro ========= ")
nuevo = np.array([[35, 13, 5000, 0, 45]])  # Age, Education-Num, Capital-gain, Capital-loss, Hours-per-week

# Aplicar la misma selección de features que usó chi2
nuevo_transformado = transformer_chi2.transform(nuevo)

# Predecir
prediccion = clf.predict(nuevo_transformado)
probabilidad = clf.predict_proba(nuevo_transformado)

etiqueta = ">50K" if prediccion[0] else "<=50K"
print(f"Salario predicho: {etiqueta}")
print(f"Probabilidad >50K : {probabilidad[0][1]*100:.1f}%")
print(f"Probabilidad <=50K: {probabilidad[0][0]*100:.1f}%")




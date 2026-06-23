import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from numpy.testing import assert_array_equal
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(BASE_DIR, "Data", "Ads")
data_filename = os.path.join(data_folder, "ad.data")

def convert_number(x):
    try:
        return float(x.strip())
    except ValueError:
        return np.nan

converters = defaultdict(convert_number)

# 1 si es anuncio, 0 si no
converters[1558] = lambda x: 1 if x.strip() == "ad." else 0

ads = pd.read_csv(data_filename, header=None, converters=converters)

X = ads.drop(1558, axis=1).values
y = ads[1558]

# Forzar conversión numérica (convierte strings residuales a NaN)
X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values

# Imputar NaN con la mediana antes de PCA
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
print("\n====== Preprocesamiento del dataset ======")
print(ads.iloc[:5, list(range(5)) + [1558]])

# Principal Component Analysis (PCA) 
pca = PCA(n_components=5)
Xd = pca.fit_transform(X)

np.set_printoptions(precision=3, suppress=True)
print("\n====== PCA ======")
print("------ Porcentaje de varianza explicada por componente ------")
print(pca.explained_variance_ratio_)

# Clasificar con features reducidas por PCA
clf = DecisionTreeClassifier(random_state=14)
scores_reduced = cross_val_score(clf, Xd, y, scoring='accuracy')

# Visualizar las dos primeras componentes
print("\n------ Visualización de primeras dos componentes ------")
classes = set(y)
colors = ['red', 'green']

for cur_class, color in zip(classes, colors):
    mask = (y == cur_class).values
    plt.scatter(Xd[mask, 0], Xd[mask, 1],
                marker='o', color=color, label=int(cur_class))

plt.legend()
plt.show()


# Nuevo transformer (desde cero)
class MeanDiscrete(TransformerMixin):
    """
    Discretiza un array NumPy usando la media como umbral.
    Valores > media -> 1, valores <= media -> 0.
    """

    def fit(self, X, y=None):
        X = as_float_array(X)
        self.mean = X.mean(axis=0)
        return self

    def transform(self, X):
        X = as_float_array(X)
        assert X.shape[1] == self.mean.shape[0]
        return X > self.mean


mean_discrete = MeanDiscrete()
X_mean = mean_discrete.fit_transform(X)


# Test del transformer con matriz nueva
def test_meandiscrete():
    print("\n====== Test de transformer MeanDiscrete() ======")
    X_test = np.array([
        [ 0,  2],
        [ 3,  5],
        [ 6,  8],
        [ 9, 11],
        [12, 14],
        [15, 17],
        [18, 20],
        [21, 23],
        [24, 26],
        [27, 29]
    ])

    mean_discrete = MeanDiscrete()
    mean_discrete.fit(X_test)

    # Verificar que las medias obtenidas sean correctas 
    assert_array_equal(mean_discrete.mean, np.array([13.5, 15.5]))
    print("------ Verificar cálculo de medias ------")
    print("Medias calculadas: ", mean_discrete.mean)
    print("Medias esperadas: ", np.array([13.5, 15.5]))

    X_transformed = mean_discrete.transform(X_test)

    X_expected = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1]
    ])

    # Verificar que la matriz resultante sea correcta 
    assert_array_equal(X_transformed, X_expected)
    print("\n------ Verificar matriz resultante ------")
    print("Matriz transformada:\n", X_transformed)
    print("Matriz esperada:\n", X_expected)


# Ejecutar el test
test_meandiscrete()


# Pipeline -> discretizar con MeanDiscrete y entrenar el clasificador
pipeline = Pipeline([
    ('mean_discrete', MeanDiscrete()),
    ('classifier', DecisionTreeClassifier(random_state=14))
])

scores_mean_discrete = cross_val_score(pipeline, X, y, scoring='accuracy')
print("\n====== Validación cruzada de pipeline ======")
print("Desempeño de Mean Discrete: {0:.3f}".format(scores_mean_discrete.mean()))


# PREDICCIÓN CON REGISTRO ARTIFICIAL
# Entrenar el pipeline con TODOS los datos (no solo cross-val)
pipeline.fit(X, y)

# --- Crear un registro falso ---
registro_anuncio = np.zeros(1558) # todo en cero
registro_anuncio[0] = 468              
registro_anuncio[1] = 60               
registro_anuncio[2] = 0.5              
registro_anuncio[3] = 1                
registro_anuncio[10] = 1             

registro_no_anuncio = np.zeros(1558) # todo en cero
registro_no_anuncio[0] = 100            
registro_no_anuncio[1] = 100            
registro_no_anuncio[2] = 0.1            

# Convertir a DataFrame
registro_anuncio_df = pd.DataFrame([registro_anuncio])
registro_no_anuncio_df = pd.DataFrame([registro_no_anuncio])

# Predicción 
pred_anuncio = pipeline.predict(registro_anuncio_df)
pred_no_anuncio = pipeline.predict(registro_no_anuncio_df)

etiquetas = {1: "Es un anuncio", 0: "NO es un anuncio"}
print("\n====== Predicción de registros artificiales ======")
print(f"Registro 1: {etiquetas[pred_anuncio[0]]}")
print(f"Registro 2: {etiquetas[pred_no_anuncio[0]]}")




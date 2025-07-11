# exportar_para_matlab.py
import pickle
import numpy as np
from sklearn.decomposition import PCA
import scipy.io
import pandas as pd

# Pesos por característica (orden según columnas de X)
feature_weights = np.array([1.0, 1.0, 1.0, 0.6, 0.6, 0.8, 0.8, 0.3, 0.7])  # ajustar según importancia

# Función para aplicar pesos
def apply_feature_weights(X):
    return X * feature_weights


# Cargar modelo
with open("svm_temblor_3s_com_extensor.pkl", "rb") as f:
    model = pickle.load(f)

# Cargar dataset
df = pd.read_csv("dataset/dataset.csv")
X = df.drop("temblor", axis=1)
y = df["temblor"]

X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
y_np = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.array(y)

# PCA a 3D
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_np)

# Calcular valores de decisión f(x) en espacio original
decision_values = model.decision_function(X_np)

# Guardar para MATLAB
scipy.io.savemat("svm_data_grid.mat", {
    "X_pca": X_pca,
    "y": y_np,
    "decision": decision_values
})
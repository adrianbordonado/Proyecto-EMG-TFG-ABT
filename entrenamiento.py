from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


# Cargar dataset
df = pd.read_csv("dataset/dataset.csv")
X = df.drop("temblor", axis=1)
y = df["temblor"]

# Pesos por característica (orden según columnas de X)
feature_weights = np.array([1.0, 1.0, 1.0, 0.6, 0.6, 0.8, 0.8, 0.3,0.7])  # ajustar según importancia

# Función para aplicar pesos
def apply_feature_weights(X):
    return X * feature_weights

# Pipeline
pipeline = Pipeline([
    ('feature_weights', FunctionTransformer(apply_feature_weights, validate=False)),
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True))
])

# Grid de hiperparámetros
param_grid = {
    'svc__C': np.linspace(7, 8, 10),
    'svc__gamma': [3.93, 5.0],
    'svc__kernel': ['rbf']
}

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# GridSearch con CV para ajustar hiperparámetros
search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=10,
    n_jobs=-1,
    verbose=1
)
search.fit(X_train, y_train)

start_time = time.time()

# Predicción en test set (sin CV)
y_pred = search.predict(X_test)
y_prob = search.predict_proba(X_test)[:, 1]  # Probabilidades para curva ROC

elapsed_time = time.time() - start_time
print(f"Tiempo de predicción: {elapsed_time:.4f} segundos")
time_per_sample = elapsed_time / len(X_test)
print(f"Tiempo medio por muestra: {time_per_sample:.6f} segundos")


# Exportar modelo entrenado
with open('svm_caca.pkl', 'wb') as f:
    pickle.dump(search, f)


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


# Reporte de clasificación
print("Best params:", search.best_params_)
print(classification_report(y_test, y_pred))

# === Matriz de Confusión ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matriz de Confusión")
plt.show()

# === Curva ROC + AUC sin CV (test set) ===
fpr_test, tpr_test, _ = roc_curve(y_test, y_prob)
roc_auc_test = auc(fpr_test, tpr_test)

# === Curva ROC + AUC con validación cruzada sobre TODO el dataset ===
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

y_probs_cv = cross_val_predict(
    search.best_estimator_, X, y, cv=cv,
    method='predict_proba', n_jobs=-1
)[:, 1]

fpr_cv, tpr_cv, _ = roc_curve(y, y_probs_cv)
roc_auc_cv = auc(fpr_cv, tpr_cv)

# === Plot ambas curvas ROC ===
plt.figure(figsize=(8,6))
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2,
         label=f'ROC sin CV (Test set) AUC = {roc_auc_test:.2f}')
plt.plot(fpr_cv, tpr_cv, color='blue', lw=2, linestyle='--',
         label=f'ROC con CV (Todo dataset) AUC = {roc_auc_cv:.2f}')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC con y sin Validación Cruzada')
plt.legend(loc="lower right")
plt.grid()
plt.show()

from sklearn.model_selection import cross_validate
# === Validación Cruzada de métricas ===
cv_results = cross_validate(search.best_estimator_, X, y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'])
# Imprimir resultados
print(f"Accuracy promedio en CV: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
print(f"Precision promedio en CV: {cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}")
print(f"Recall promedio en CV: {cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}")
print(f"F1 promedio en CV: {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")


# Exportar modelo entrenado
with open('modelos/svm_temblor_3s.pkl', 'wb') as f:
    pickle.dump(search, f)



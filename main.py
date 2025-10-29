# ========================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ========================================
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ========================================
# 2. CARGA O DESCARGA DE DATOS
# ========================================
file_path = "Telco-Customer-Churn.csv"
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


# Si el archivo no existe, se descarga automáticamente
if not os.path.exists(file_path):
    print("📥 Descargando dataset Telco Customer Churn...")
    df = pd.read_csv(url)
    df.to_csv(file_path, index=False)
    print("✅ Dataset descargado y guardado como:", file_path)
else:
    print("✅ Cargando dataset local...")
    df = pd.read_csv(file_path)

print("\nPrimeras filas del dataset:")
print(df.head())
print("\nInformación general:")
print(df.info())

# ========================================
# 3. LIMPIEZA DE DATOS
# ========================================

# Eliminar columnas irrelevantes o con muchos valores únicos
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Convertir valores no numéricos en numéricos cuando sea necesario
if df['TotalCharges'].dtype == 'object':
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Llenar valores faltantes con la media
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

# ========================================
# 4. ANÁLISIS EXPLORATORIO (Opcional)
# ========================================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title("Distribución de Churn (Clientes que abandonan)")
plt.show()

# ========================================
# 5. ENCODING DE VARIABLES CATEGÓRICAS
# ========================================
cat_cols = df.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in cat_cols:
    if df[col].nunique() == 2:
        df[col] = le.fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# ========================================
# 6. SEPARAR VARIABLES INDEPENDIENTES Y DEPENDIENTES
# ========================================
X = df.drop('Churn', axis=1)
y = df['Churn']

# ========================================
# 7. DIVISIÓN ENTRE TRAIN Y TEST
# ========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalamos las variables numéricas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========================================
# 8. ENTRENAMIENTO DE MODELOS
# ========================================
models = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, preds))
    print("Reporte de clasificación:")
    print(classification_report(y_test, preds))

# ========================================
# 9. COMPARACIÓN DE MODELOS
# ========================================
plt.figure(figsize=(6,4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Comparación de Accuracy entre modelos")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.0)
plt.show()

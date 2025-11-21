# Análisis y Preprocesamiento de Datos
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1.2. Cargar el dataset original
df = pd.read_csv("Telco-Customer-Churn.csv")

# 1.3. Revisar estructura
print(df.info())         # Tipos de datos
print(df.describe())     # Estadísticas básicas
print(df.head())         # Primeras filas

# 1.4. Buscar valores nulos
print(df.isnull().sum())

# Si hay valores nulos, se eliminan (o se imputan)
df = df.dropna()

# 1.5. Distribución de la variable objetivo
sns.countplot(x='Churn', data=df)
plt.title("Distribución de Clientes (Fuga vs No Fuga)")
plt.show()

# 1.6. Relación entre Churn y MonthlyCharges
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Relación entre Churn y MonthlyCharges")
plt.show()

# Guardar dataset limpio
df.to_csv("datos_limpios.csv", index=False)
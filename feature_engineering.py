# Feature Engineering y codificación
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2.1. Cargar dataset limpio
df = pd.read_csv("datos_limpios.csv")

# 2.2. Codificar variables categóricas
df = pd.get_dummies(df, drop_first=True)

# 2.3. Separar variables predictoras y objetivo
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# 2.4. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2.5. Escalar variables numéricas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

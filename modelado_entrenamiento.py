# 👩‍💻 3️⃣ Modelado y Entrenamiento
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Modelos
def entrenar_modelos(X_train, y_train, X_test, y_test):
    resultados = {}

    # Regresión Logística
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    resultados['Logistic Regression'] = accuracy_score(y_test, pred_lr)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    resultados['Random Forest'] = accuracy_score(y_test, pred_rf)

    # Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    pred_gb = gb.predict(X_test)
    resultados['Gradient Boosting'] = accuracy_score(y_test, pred_gb)

    return resultados, rf, y_test, pred_rf
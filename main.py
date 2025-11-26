# script principal que une todo
from feature_engineering import X_train, X_test, y_train, y_test
from modelado_entrenamiento import entrenar_modelos
from evaluacion_visualizacion import visualizar_resultados

# Entrenar y evaluar
resultados, modelo_rf, y_test, pred_rf = entrenar_modelos(X_train, y_train, X_test, y_test)
visualizar_resultados(resultados, y_test, pred_rf)

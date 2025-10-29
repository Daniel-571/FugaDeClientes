# 👨‍💻 4️⃣ Evaluación y Visualización Final
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def visualizar_resultados(resultados, y_test, pred_rf):
    # 4.1. Comparar métricas
    modelos = list(resultados.keys())
    accuracy = list(resultados.values())

    plt.bar(modelos, accuracy, color=['skyblue', 'green', 'orange'])
    plt.title("Comparación de Accuracy entre Modelos")
    plt.ylabel("Accuracy")
    plt.show()

    # 4.2. Matriz de confusión
    cm = confusion_matrix(y_test, pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión - Random Forest")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # 4.3. Reporte detallado
    print(classification_report(y_test, pred_rf))
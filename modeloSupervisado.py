#Librerías necesarias
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

#Para ignoarar algunas sugerencias que me salian
warnings.filterwarnings("ignore")

#Lee el archivo .CSV
df = pd.read_csv('transporte.csv', sep=';')


#Codifica las variables categóricas
le_origen = LabelEncoder()
le_destino = LabelEncoder()
le_hora = LabelEncoder()
le_vehiculo = LabelEncoder()

df['origen'] = le_origen.fit_transform(df['origen'])
df['destino'] = le_destino.fit_transform(df['destino'])
df['hora-dia'] = le_hora.fit_transform(df['hora-dia'])
df['vehiculo'] = le_vehiculo.fit_transform(df['vehiculo'])


#Variables predictoras y variable objetivo
X = df[['origen', 'destino', 'hora-dia', 'vehiculo', 'tiempo-viaje', 'pasajeros']]
y = df['retraso']

#División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Crea y entrenar el modelo
modelo = DecisionTreeClassifier(max_depth=4)
modelo.fit(X_train, y_train)

#Hace las predicciones
y_pred = modelo.predict(X_test)

#Calcula las métricas
precision = accuracy_score(y_test, y_pred)
reporte = classification_report(y_test, y_pred, output_dict=True)


#Mostra precisión general
print(f"Precisión general del modelo: {precision:.2f}\n")

#Mostra reporte traducido
print("REPORTE DE CLASIFICACIÓN:\n")
print(f"{'Clase':<10}{'Precisión':<12}{'Exhaustividad':<16}{'F1-score':<12}{'Cantidad'}")

for clase, valores in reporte.items():
    if clase in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    print(f"{clase:<10}{valores['precision']:<12.2f}{valores['recall']:<16.2f}{valores['f1-score']:<12.2f}{int(valores['support'])}")


#Visualización del árbol de decisión
plt.figure(figsize=(20, 12))
plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=modelo.classes_,
    filled=True,
    fontsize=12
)
plt.title("Árbol de Decisión - Predicción de Retraso", fontsize=18)
plt.show()

# Machine_learning
Para este ejemplo se utiliza el conjunto de datos Iris.
El conjunto de datos contiene 50 muestras de cada una de tres especies de Iris (Iris setosa, Iris virginica e Iris versicolor). Se midió cuatro rasgos de cada muestra: el largo y ancho del sépalo y pétalo, en centímetros.
Se propone un ejemplo sencillo de entrenamiento, evalución y predicción.
```
# Importar las librerías necesarias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
```
A continuación se cargan el conjunto de datos IRIS.
Además se divide el conjunto de datos en dos subconjuntos para el entrenamiento.
```
# Cargar el conjunto de datos iris
iris = load_iris()

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
```
Seguidamente se genera un modelo. Para ello se elige un algoritmo para el entrenamiento.
En éste caso se ha elegido una Regeresión Logística con 1000 iteraciones.
Las iteraciones ayudan a la convergencia del modelo. Se pueden probar diferentes algoritmos y valores de iteración.
```
# Crear un modelo de regresión logística
model = LogisticRegression(max_iter=1000)
```
El siguiente paso es el entrenamiento del modelo.
Para ello le pasamos los datos para entrenar.
```
# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)
```
Para ver como funciona, le pasamos los datos de test.
No han sido procesados. Por lo tanto será indicador de lo bueno o malo que es 
el modelo en las predicciones.
```
# Realizar predicciones en los datos de prueba
y_pred = model.predict(X_test)
# Define the class names
class_names = ['setosa', 'versicolor', 'virginica']
```
Para saber y visualizar la capacidad de predicción
le pasamos un neuvo ejemplo para que realize una predicción.
```
# Generar una predicción con un nuevo ejemplo.
new_sample = np.array([[5.0, 3.4, 1.5, 0.2]])
probabilities = model.predict_proba(new_sample)[0]
predicted_class = np.argmax(probabilities)
```
Para finaliza se imprime el resulatdo de la predicción así
como la precisión.
```
# Imprimir el resultado de la predicción
print(f"The predicted class is {class_names[predicted_class]} with a probability of {probabilities[predicted_class]:.2f}.")

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Imprimir la precisión del modelo
print("Precisión del modelo: {:.2f}%".format(accuracy * 100))
```

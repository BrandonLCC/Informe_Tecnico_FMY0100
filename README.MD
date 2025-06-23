# Caso - Counter Strike

Empresa: Valve

Versión del proyecto: Counter Strike: GO

Integrantes: Brandon Casas, Felipe Godoy, Leo

Docente: Giocrisrai Godoy Bonillo

## Etapa 1  

## Etapa 2 - Modelos de regresión 

### Modelos 

- Simple Linear Regression Model
- Multiple Linear Regression Model
- Decision Tree Model
- Support Vector Regression (SVR) Model
- Random Forest Model

### Metricas para modelos de regresión

Estas son las metricas que se utlizaran para los modelos de regresión. Estas métricas nos permitira cuantificar la calidad de las predicciones de los modelos.

- Mean Squared Error (MSE): error cuadrático medio 
- Mean Absolute Error (MAE): error absoluto medio 
- Root Mean Squared Error (RMSE): raíz del error cuadrático medio
- R2 Score: coeficiente de determinación o R al cuadrado

## Etapa 3 - Modelos de clasificación

### Modelos 

- Logistic Regression Model
- KNN (K-Nearest Neighbors) Model
- SVC (Support Vector Classification) Model
- Decision Tree classification Model
- Random Forest Classification Model

### Metricas para modelos de clasificación

- ACCURACY (exactitud) 
- ERROR RATE (tasa de error)
- SENSITIVITY o RECALL (sensibilidad)
- SPECIFITY (especificidad)
- F1 SCORE : (Resumen de la precisión y sensibilidad en una sola métrica).
- Curva ROC (Eeceiver Operating Characteristic) 
- Curva PR  (Precision-Recall)

### Mejor modelo de clasificación

De todos los modelos realizados, el Random Forest Classifier obtuvo el mejor desempeño en la predicción de partidas ganadas y perdidas. Utilizando las características *TeamStartingEquipmentValue, MatchKills, MatchAssists y Map * como variables, se obtuvieron los siguientes resultados: para la clase 0 se alcanzó una precisión de 0.68, un recall de 0.63 y un f1-score de 0.66, lo que representa un rendimiento aceptable. Para la clase 1, se logró una precisión de 0.62, un recall de 0.67 y un f1-score de 0.64.
El accuracy o exactitud del modelo fue de 0.65, o un 65% de predicciones correctas considerando todos los valores de la matriz de confusión, la cual fue la siguiente:

Matriz de Confusión:

[[5244 3067]

[2449 4962]]

Sensitivity o la cantidad de casos positivos que el algorimo ha clasificado correctamente con un valor de: 0.6695

Specificity o la cantidad de casos negativos que el algorimo ha clasificado correctamente con un valor de: 0.6310

Curva ROC: 0.71

Curva PR: 0.68

En el uso del GridsearchCV para encontrar los mejores hiperparametros, Se obtuvo para la clase 0 una precision de 0.66, un recall de 0.67 y un f1-score de 0.66. Para la clase 1 una precision de 0.62 un recall de 0.61 y uun f1-score de 0.62. Siendo estos resultados un poco menor que el resultado del primer modelo.


# Entrega final



# Como contribuir al proyecto

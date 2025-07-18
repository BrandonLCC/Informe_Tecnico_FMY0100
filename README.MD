
![Presentación](https://raw.githubusercontent.com/BrandonLCC/Informe_Tecnico_FMY0100/main/images/imagen-presentacion.webp)

# Caso - Counter Strike: Global Offensive

Empresa: Valve

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

### Mejor modelo de regresión
  
Resultados del Modelo de Regresión Lineal Simple

Mean Squared Error (MSE): 648585.7053669427

Mean Absolute Error (MAE): 511.28882520725983

Root Mean Squared Error (RMSE): 805.3481889015103

R2 Score: 0.8541586266087651

observacion:

El modelo muestra un desempeño robusto, explicando aproximadamente el 85.4% de la variación de la variable dependiente. El MAE indica un error promedio de 511 unidades entre las predicciones y los valores reales, mientras que el RMSE de 805 refleja la presencia de errores algo más pronunciados, aunque dentro de un rango aceptable. En general, se considera un modelo con buen ajuste para las variables seleccionadas.

resultado del GridseahCV

Mean Squared Error (MSE): 648581.7053669427

Mean Absolute Error (MAE): 511.28882520725983

Root Mean Squared Error (RMSE): 805.3481889015103

R2 Score: 0.8541586266087651

Observacion:

con GridSearchCV para optimizar los hiperparámetros del modelo, los resultados obtenidos se mantienen iguales solo hubo un cambio pequeño en MSE.

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



# Como instalar el proyecto

Desde tu terminal de visual estudio sigue estos siguientes pasos para instalar y probar el proyecto

1. Crea tu entorno virtual

``` python -m venv venv ```

2. Activa el entorno virtual

``` venv\Scripts\activate ```

2. Realizar un pip install

``` pip install -r requirements.txt ```

3. Ingresar a la carpeta del  poryecto 

``` cd proyecto_csgo ```

4. Entrenar modelo

``` entrenar_modelo.py ```

5. Ejecuta el simulador de prediciones

``` uvicorn main:app --reload ```

6. Selecciona la tecla ctr+click

``` http://127.0.0.1:8000 ```

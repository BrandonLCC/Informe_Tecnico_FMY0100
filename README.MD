
![Presentación](https://raw.githubusercontent.com/BrandonLCC/Informe_Tecnico_FMY0100/main/images/imagen-presentacion.webp)

# Caso - Counter Strike: Global Offensive

Empresa: Valve

Integrantes: Brandon Casas, Felipe Godoy, Leo

Docente: Giocrisrai Godoy Bonillo

## Etapa 1  

En esta primera etapa, nos centramos en las dos fases inicales de la metodologia CRISP-DM: Business Understanding y la Fase 2: Data Understanding. La cuales son fundamentales para orientar nuestra investigación.

En la fase Business Understanding, buscamos comprender el problema desde el punto de vista empresarial, definiendo claramente el objetivo del proyecto para luego realizar las tareas de analisis de datos.

En la Fase de Data Understanding se realizan las primeras actividades realiadas con el conjunto de datos, lo que nos permite:

- Identificar y familiarizarnos con las variables disponibles.
- Determinar la naturaleza de los datos, distinguiendo entre variables numéricas y categóricas.
- Detectar valores duplicados o datos nulos. 
- Analizar el comportamiento de los datos mediante herramientas gráficas como diagramas de dispersión, gráficos de barras, histogramas, entre otros.
- Evaluar la correlación entre variables para identificar posibles relaciones significativas.

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

Mean Squared Error (MSE): 645929.0597126659

Mean Absolute Error (MAE): 509.20139631339185

Root Mean Squared Error (RMSE): 803.6971193880602

R2 Score:  0.8556029403163038

observacion:

El modelo muestra un desempeño robusto, explicando aproximadamente el 85.5% de la variación de la variable dependiente. El MAE indica un error promedio de 509 unidades entre las predicciones y los valores reales, mientras que el RMSE de 803 refleja la presencia de errores algo más pronunciados, aunque dentro de un rango aceptable. En general, se considera un modelo con buen ajuste para las variables seleccionadas.

resultado del GridseahCV

Mean Squared Error (MSE): 645927.1032691925

Mean Absolute Error (MAE): 509.1785189644252

Root Mean Squared Error (RMSE): 803.695902234914

R2 Score: 0.8556033776780865

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

De todos los modelos realizados, el Random Forest Classifier obtuvo el mejor desempeño en la predicción de partidas ganadas y perdidas. Utilizando las características *MatchKills, MatchAssists, MatchHeadshots, RoundStartingEquipmentValue, RNonLethalGrenadesThrown, Team y  Map* como variables, se obtuvieron los siguientes resultados: para la clase 0 se alcanzó una precisión de  0.71 , un recall de 0.70 y un f1-score de  0.71, lo que representa un rendimiento aceptable. Para la clase 1, se logró una precisión de   0.71, un recall de 0.71 y un f1-score de 0.71.
El accuracy o exactitud del modelo fue de 0.71 , o un 71 % de predicciones correctas considerando todos los valores de la matriz de confusión, la cual fue la siguiente:

Matriz de Confusión:

![Presentación](https://raw.githubusercontent.com/BrandonLCC/Informe_Tecnico_FMY0100/main/images/matriz.png)

Sensitivity o la cantidad de casos positivos que el algorimo ha clasificado correctamente con un valor de: 0.7140

Specificity o la cantidad de casos negativos que el algorimo ha clasificado correctamente con un valor de:  0.7052

Curva ROC: 0.79

Curva PR: 0.78

En el uso del GridsearchCV para encontrar los mejores hiperparametros, Se obtuvo para la clase 0 una precision de 0.70, un recall de 0.75 y un f1-score de 0.72. Para la clase 1 una precision de 0.73 un recall de  0.69 y uun f1-score de  0.71.


# Entrega final

- Mejora y optimización de los modelos de regresión y clasificación
- Deployment del proyecto
- Mejora en los notebooks

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

# limpieza_entrenamiento.py
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from imblearn.over_sampling import SMOTE
# 1. Cargar dataset
df = pd.read_csv("data/Anexo ET_demo_round_traces_2022 .csv", sep=";")  # Note the space before .csv


# 2. Copia de respaldo
df_backup = df.copy()

# 3. Limpieza
df_backup.drop(columns=['Unnamed: 0', 'AbnormalMatch', 'FirstKillTime', 'TimeAlive', 'TravelledDistance'], inplace=True)
df_backup.dropna(inplace=True)
df_backup = df_backup[df_backup['MatchKills'] <= 28]
df_backup = df_backup[df_backup['MatchAssists'] <= 8]
df_backup = df_backup[(df_backup['RoundId'] >= 1) & (df_backup['RoundId'] <= 30)]

# 4. Transformaciones
le = LabelEncoder()
df_backup['Team'] = le.fit_transform(df_backup['Team'])
df_backup['Map'] = le.fit_transform(df_backup['Map'])

df_backup['RoundWinner'] = df_backup['RoundWinner'].astype(bool).replace({True: 1, False: 0})
df_backup['MatchWinner'] = df_backup['MatchWinner'].astype(bool).replace({True: 1, False: 0})
df_backup['Survived'] = df_backup['Survived'].astype(bool).replace({True: 1, False: 0})

# 5. Modelo de regresión
X_reg = df_backup[['TeamStartingEquipmentValue']]
y_reg = df_backup[['RoundStartingEquipmentValue']]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.30, random_state=42)

modelo_reg = LinearRegression()
modelo_reg.fit(X_train_reg, y_train_reg)

# Guardar modelo regresión
joblib.dump(modelo_reg, "models/modelo_regresion.pkl")
joblib.dump(X_reg.columns.tolist(), "models/columnas_regresion.pkl")

# 6. Modelo de clasificación


X_clf= df_backup[['MatchKills','MatchAssists','MatchHeadshots','RoundStartingEquipmentValue','RNonLethalGrenadesThrown','Team','Map']]
y_clf= df_backup['RoundWinner']

smote = SMOTE(random_state=42)
X_clf_resampled, y_clf_resampled = smote.fit_resample(X_clf, y_clf)


# Dividir en train/test con datos balanceados o sin balancear si no se pudo aplicar SMOTE
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf_resampled, y_clf_resampled, test_size=0.20, random_state=42
)
modelo_clf = RandomForestClassifier(    
       max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight={0: 1, 1: 2.9},
    random_state=42,
    n_jobs=-1
    )

modelo_clf.fit(X_train_clf, y_train_clf)

print("Modelo de clasificación ")
y_pred = modelo_clf.predict(X_test_clf)
print(classification_report(y_test_clf, y_pred))


#ver metrica de regresion mae

print("Modelo de regresión:")
from sklearn.metrics import mean_squared_error, r2_score 
y_pred_reg = modelo_reg.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))
print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
print("RMSE:", rmse)
print("R2:", r2_score(y_test_reg, y_pred_reg))
print("✅ Modelos entrenados y guardados con éxito (Regresión Lineal y RandomForestClassifier).")

# Guardar modelo clasificación
joblib.dump(modelo_clf, "models/modelo_clasificacion.pkl")
joblib.dump(X_clf.columns.tolist(), "models/columnas_clasificacion.pkl")
print("✅ Modelos entrenados y guardados con éxito ( RandomForestClassifier y Regresión).")




# URL base de tu API (cambia el puerto si usas otro)
import requests


url_base = "http://localhost:8000"

# Datos de prueba para regresión
datos_regresion = {
    "TeamStartingEquipmentValue": 3000.0
}

# Datos de prueba para clasificación
datos_clasificacion = {
    "Map": 1,
    "Team": 2,
    "RoundStartingEquipmentValue": 3500.0
}

def test_regresion():
    resp = requests.post(f"{url_base}/predict/regresion", json=datos_regresion)
    if resp.status_code == 200:
        print("Predicción regresión:", resp.json())
    else:
        print("Error regresión:", resp.text)

def test_clasificacion():
    resp = requests.post(f"{url_base}/predict/clasificacion", json=datos_clasificacion)
    if resp.status_code == 200:
        print("Predicción clasificación:", resp.json())
    else:
        print("Error clasificación:", resp.text)

if __name__ == "__main__":
    test_regresion()
    test_clasificacion()

# ver metricas de clasificacion
def test_metricas_clasificacion():
    resp = requests.get(f"{url_base}/metrics/clasificacion")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Simulador CSGO")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

modelo_regresion = joblib.load("models/modelo_regresion.pkl")
modelo_clasificacion = joblib.load("models/modelo_clasificacion.pkl")
columnas_clasificacion = joblib.load("models/columnas_clasificacion.pkl")

class InputRegresion(BaseModel):
    TeamStartingEquipmentValue: float

class InputClasificacion(BaseModel):
    MatchKills: int
    MatchAssists: int
    MatchHeadshots: int
    RoundStartingEquipmentValue: float
    RNonLethalGrenadesThrown: int  # ajusta al nombre correcto
    Team: int
    Map: int

@app.post("/predict/regresion")
def predict_regresion(data: InputRegresion):
    try:
        entrada = pd.DataFrame(
            [[data.TeamStartingEquipmentValue]],
            columns=["TeamStartingEquipmentValue"]
        )
        prediccion = modelo_regresion.predict(entrada)
        valor_total = float(prediccion[0])

        num_jugadores = 5
        valor_por_jugador = valor_total / num_jugadores

        # Clasificar tipo de ronda según valor total del equipo
        if valor_total <= 5000:
            ronda = "ECO"
            recomendacion = "Considera pistolas básicas o guardar dinero"
        elif valor_total <= 12500:
            ronda = "Semi-eco"
            recomendacion = "Compra limitada, como pistolas fuertes o SMGs con algunas granadas"
        elif valor_total <= 20000:
            ronda = "Force buy"
            recomendacion = "Compra forzada. Armas medias sin armadura completa o sin utilidad"
        else:
            ronda = "Full buy"
            recomendacion = "Compra completa con rifles, armadura y granadas"

        return {
            "RoundStartingEquipmentValue_predicho": round(valor_total, 2),
            "Valor_estimado_por_jugador": round(valor_por_jugador, 2),
            "Tipo_de_ronda": ronda,
            "Recomendacion": recomendacion
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción de regresión: {str(e)}")

@app.post("/predict/clasificacion")
def predict_clasificacion(data: InputClasificacion):
    try:
        entrada = pd.DataFrame(
            [[
                data.MatchKills,
                data.MatchAssists,
                data.MatchHeadshots,
                data.RoundStartingEquipmentValue,
                data.RNonLethalGrenadesThrown,
                data.Team,
                data.Map
            ]],
            columns=columnas_clasificacion
        )
        prediccion = modelo_clasificacion.predict(entrada)
        return {"RoundWinner_predicho": int(prediccion[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción de clasificación: {str(e)}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from catboost import CatBoostClassifier
import pickle
import logging
import classes
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStore:
    def __init__(self):
        self.model = None
        self.pipeline = None

    def load(self):
        """
        Загружаем модели из файлов
        """
        logger.info("Loading model and pipeline...")
        # Load CatBoost model
        self.model = CatBoostClassifier()
        self.model.load_model("artifacts/catboost_model.bin")

        # Load preprocessing pipeline
        with open("artifacts/pipeline.pkl", "rb") as f:
            self.pipeline = pickle.load(f)

        logger.info("Model and pipeline loaded successfully!")

    def predict(self, input_data: pd.DataFrame) -> List[List[int]]:
        """
        Трансформирует исходные данные и возвращает предсказание
        """
        processed_data = self.pipeline.transform(input_data)

        predictions = self.model.predict(processed_data)
        return predictions.tolist()

feature_store = FeatureStore()

PRODUCT_MAP = {
    "ind_ahor_fin_ult1": "Сберегательный счёт",
    "ind_aval_fin_ult1": "Банковская гарантия",
    "ind_cco_fin_ult1": "Текущие счета",
    "ind_cder_fin_ult1": "Деривативный счёт",
    "ind_cno_fin_ult1": "Зарплатный проект",
    "ind_ctju_fin_ult1": "Детский счёт",
    "ind_ctma_fin_ult1": "Особый счёт 3",
    "ind_ctop_fin_ult1": "Особый счёт",
    "ind_ctpp_fin_ult1": "Особый счёт 2",
    "ind_deco_fin_ult1": "Краткосрочный депозит",
    "ind_deme_fin_ult1": "Среднесрочный депозит",
    "ind_dela_fin_ult1": "Долгосрочный депозит",
    "ind_ecue_fin_ult1": "Цифровой счёт",
    "ind_fond_fin_ult1": "Денежный средства",
    "ind_hip_fin_ult1": "Ипотека",
    "ind_plan_fin_ult1": "Пенсионный план",
    "ind_pres_fin_ult1": "Кредит",
    "ind_reca_fin_ult1": "Налоговый счёт",
    "ind_tjcr_fin_ult1": "Кредитная карта",
    "ind_valo_fin_ult1": "Ценные бумаги",
    "ind_viv_fin_ult1": "Домашний счёт",
    "ind_nomina_ult1": "Аккаунт для выплаты зарплаты",
    "ind_nom_pens_ult1": "Аккаунт для пенсионных обязательств",
    "ind_recibo_ult1": "Дебетовый аккаунт",
}

# Интерфейсы
class ClientData(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    predicted_products: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    feature_store.load()
    logger.info("Application is ready!")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """
    Health check
    """
    return {"status": "ok"}    

@app.post("/predict", response_model=List[PredictionResponse])
def predict(data: List[ClientData]):
    """
    Возвращает предсказание банковских услуг на основе данных активности клиента
    """
    try:
        # Convert input data to DataFrame
        raw_input_data = pd.DataFrame([d.features for d in data])

        # Make predictions
        predictions = feature_store.predict(raw_input_data)

        # Map predictions to product names
        responses = []
        for row in predictions:
            predicted_products = [
                PRODUCT_MAP[product]
                for product, value in zip(PRODUCT_MAP.keys(), row)
                if value == 1
            ]
            responses.append({"predicted_products": predicted_products})

        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

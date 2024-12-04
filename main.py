from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from requests import Request
from typing import List, Optional
import pandas as pd
from catboost import CatBoostClassifier
import pickle
import logging
import time
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Класс взаимодействия с моделью
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
        processed_data = pd.DataFrame(processed_data, columns=['age', 'antiguedad',
            'renta', 'fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo',
            'fecha_alta', 'ind_nuevo', 'indrel', 'indrel_1mes', 'tiprel_1mes', 'indresi',
            'indext', 'canal_entrada', 'indfall', 'cod_prov', 'ind_actividad_cliente', 'segmento'])
        predictions = self.model.predict(
            processed_data[["ncodpers", "fecha_dato", "age", "renta", "sexo"]])

        return predictions.tolist()

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
    fecha_dato: str
    ncodpers: int
    ind_empleado: Optional[str]
    pais_residencia: Optional[str]
    sexo: Optional[str]
    age: Optional[str]
    fecha_alta: Optional[str]
    ind_nuevo: Optional[int]
    antiguedad: Optional[str]
    indrel: Optional[float]
    ult_fec_cli_1t: Optional[str]
    indrel_1mes: Optional[str]
    tiprel_1mes: Optional[str]
    indresi: Optional[str]
    indext: Optional[str]
    conyuemp: Optional[str]
    canal_entrada: Optional[str]
    indfall: Optional[str]
    tipodom: Optional[float]
    cod_prov: Optional[float]
    nomprov: Optional[str]
    ind_actividad_cliente: Optional[float]
    renta: Optional[float]
    segmento: Optional[str]


class PredictionResponse(BaseModel):
    predicted_products: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    feature_store.load()
    logger.info("Application is ready!")
    yield

app = FastAPI(lifespan=lifespan)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

feature_store = FeatureStore()

# Метрика задержки запросов
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'Histogram for the duration in seconds for request latency',
    ['method', 'endpoint', 'status_code']
)

# Метрика количества запросов
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Counter for total requests to the app',
    ['method', 'endpoint', 'status_code']
)

# Метрика количества вызовов энпоинта с предсказаниями
PREDICTION_COUNT = Counter(
    'prediction_requests_total', 
    'Counter for total prediction requests',
    ['model_version', 'status']
)

# Снимаем общие метрики для всех запросов через специальный middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    REQUEST_LATENCY.labels(request.method, request.url.path, response.status_code).observe(process_time)
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    return response

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
async def health_check():
    """
    Health check
    """
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: ClientData):
    """
    Возвращает предсказание банковских услуг на основе данных активности клиента
    """
    model_version = "1.0"

    try:
        input_dict = data.dict()
        raw_input_data = pd.DataFrame([input_dict])

        predictions = feature_store.predict(raw_input_data)

        predicted_products = [
            PRODUCT_MAP[product]
            for product, value in zip(PRODUCT_MAP.keys(), predictions[0])
            if value == 1
        ]

        PREDICTION_COUNT.labels(model_version, "success").inc()

        return {"predicted_products": predicted_products}
    except Exception as e:
        PREDICTION_COUNT.labels(model_version, "failure").inc()

        raise HTTPException(status_code=500, detail=str(e)) from e

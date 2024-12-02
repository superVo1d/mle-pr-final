import pendulum
from airflow.decorators import dag, task
import pandas as pd
import os
import joblib

ORIGINAL_DATA_PATH = "data/train_ver2.csv"
TMP_DATA_PATH = "data/tmp_data.parquet"
TRANSFORMED_DATA_PATH = "data/data_transformed.parquet"
PIPELINE_PATH = "artifacts/pipeline.joblib"

@dag(
    schedule='@once',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ETL"],
)
def preprocess_dataset():
    @task()
    def load_data():
        """Загружает данные из CSV"""
        data = pd.read_csv(ORIGINAL_DATA_PATH, low_memory=False)

        os.makedirs(os.path.dirname(TMP_DATA_PATH), exist_ok=True)
        data.to_parquet(TMP_DATA_PATH, index=False)

        return TMP_DATA_PATH

    @task()
    def prepare_data(file_path: str):
        """Загружает пайплайн, подготавливает и сохраняет данные"""
        data = pd.read_parquet(file_path)
        pipeline = joblib.load(PIPELINE_PATH)

        processed_data = pipeline.transform(data)
        feature_names = pipeline.named_steps['feature_transformation'].get_feature_names_out()
        data_transformed = pd.DataFrame(processed_data, columns=feature_names)

        os.makedirs(os.path.dirname(TRANSFORMED_DATA_PATH), exist_ok=True)
        data_transformed.to_parquet(TRANSFORMED_DATA_PATH)

    intermediate_file_path = load_data()
    prepare_data(intermediate_file_path)

preprocess_dataset()

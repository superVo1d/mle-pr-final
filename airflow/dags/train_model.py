import pendulum
from airflow.decorators import dag, task
import pandas as pd
from catboost import CatBoostClassifier
from catboost import Pool

TRANSFORMED_DATA_PATH = "data/data_transformed.parquet"
MODEL_PATH = "artifacts/catboost_model.bin"

@dag(
    schedule='@once',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ETL"],
)
def train_model():
    @task()
    def load_and_train():
        """Загружает данные, заново обучает модель"""
        data = pd.read_parquet(TRANSFORMED_DATA_PATH)

        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)

        product_cols = [col for col in data.columns if col.endswith("_ult1")]

        interactions = data[["ncodpers", "fecha_dato", "age", "renta", "sexo"] + product_cols]

        train_test_global_time_split_date = pd.Timestamp("2016-01-01")

        train_test_global_time_split_idx = interactions["fecha_dato"] < train_test_global_time_split_date
        interactions_train = interactions[train_test_global_time_split_idx]

        X_train = interactions_train.drop(columns=['fecha_dato', 'ncodpers'] + product_cols)
        y_train = interactions_train[product_cols]

        X_train_sample = X_train.sample(frac=0.1, random_state=42) # Для упрощения работы оставим 10% данных, иначе падает ядро
        y_train_sample = y_train.loc[X_train_sample.index]

        train_pool = Pool(
            data=X_train_sample,
            label=y_train_sample,
            # cat_features=['segmento']
        )

        model.fit(train_pool)

        model.save_model(MODEL_PATH)

    load_and_train()


train_model()

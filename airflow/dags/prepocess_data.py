import pendulum
from airflow.decorators import dag, task
import pandas as pd
import os
import joblib

ORIGINAL_DATA_PATH = "data/train_ver2.csv"
TRANSFORMED_DATA_PATH = "data/data_transformed.parquet"
PIPELINE_PATH = "artifacts/training_pipeline.pkl"

@dag(
    schedule='@once',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ETL"],
)
def preprocess_dataset():
    @task()
    def prepare_data():
        """Загружает пайплайн, подготавливает и сохраняет данные"""
        data = pd.read_csv(ORIGINAL_DATA_PATH, low_memory=False)
        pipeline = joblib.load(PIPELINE_PATH)

        processed_data = pipeline.transform(data)
        processed_data = pd.DataFrame(processed_data, columns=
            ['age', 'antiguedad', 'renta', 'fecha_dato', 'ncodpers', 'ind_empleado',
            'pais_residencia', 'sexo', 'fecha_alta', 'ind_nuevo', 'indrel', 'indrel_1mes',
            'tiprel_1mes', 'indresi', 'indext', 'canal_entrada', 'indfall', 'cod_prov',
            'ind_actividad_cliente', 'segmento', 'ind_ahor_fin_ult1',
            'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
            'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1',
            'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
            'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1',
            'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
            'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',
            'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1',
            'ind_nom_pens_ult1', 'ind_recibo_ult1'])

        os.makedirs(os.path.dirname(TRANSFORMED_DATA_PATH), exist_ok=True)
        processed_data.to_parquet(TRANSFORMED_DATA_PATH)

    prepare_data()

preprocess_dataset()

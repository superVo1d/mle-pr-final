# Рекомендации банковских продуктов

Цель: предсказать, какими финансовыми услугами клиенты банка воспользуются в следующем месяце, исходя из их характеристик и истории поведения. Это поможет улучшить таргетинг предложений, оптимизировать маркетинговые затраты и повысить удовлетворенность клиентов.

Основные задачи: 
1. Анализ данных о клиентах (EDA) для выявления закономерностей и особенностей.
2. Определение ключевых метрик успеха с учетом бизнес-целей.
3. Моделирование поведения клиентов с использованием методов машинного обучения.
4. Продуктивизация модели и интеграция в бизнес-процессы.
5. Настройка мониторинга качества модели и процессов дообучения.

## Основные метрики

Для задачи многофакторной классификации (multi-label classification), с учетом специфики бизнес-задачи, ключевыми метриками будут:
- F1-Score
- Precision
- Recall

## Запуск проекта

### Настрока окружения 
```
# Скачиваем репозиторий
git clone https://github.com/superVo1d/mle-pr-final
cd mle-pr-final

# Устанавиливаем окружение
python3 -m venv .venv
source .venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt

# Создаем файл с перменными окружения, заполняем его по примеру .env.example
touch .env
```

### Запуск MLFlow

Для запуска MLFlow с удаленным хранилищем вртефактов необходимо выполнить скрипт:
```
sh run_mlflow_locally.sh
```

Для запуска MLFlow локально необходимо выполнить скрипт:
```
sh run_mlflow_locally.sh
```

После чего интерфейс MLFlow будет доступен по адресу ```http://127.0.0.1:5000```.

### Загрузка данных

Скачиваем данные ```https://disk.yandex.com/d/Io0siOESo2RAaA``` в  директорию ```data```. 

### Запуск ноутбука с исследованием данных

```
jupyter lab EDA.ipynb
```

Выбранная на этой стадии модель обучена на исторических данных клиентов банка за 2015 год и способна предсказывать для данного клиента выбранные им услуги в следующем периоде.

### Запуск ноутбука с моделированием и препроцессингом данных

```
jupyter lab modeling.ipynb
```

На этом этапе создается два паплайна для обработки данных:
1. artifacts/training_pipeline.pkl — пайплайн для обучения модели. В нем дополнительно удалены выбросы и присутствует обработка таргетов.
2. artifacts/pipeline.pkl — паплайн, который будет использоваться в продакшн среде.

Также здесь обучается и сохраняется модель многофакторной классификации CatBoost. Артефакты сохраняются в Mlflow для воспроизводимости экспериментов.

### Airflow

Для переобучения модели по расписанию на исторических данных развернем Airflow:
```
docker compose up --build
```

После чего интерфейс Airflow будет доступен по адресу ```http://127.0.0.1:8080```.

Для переобучения модели сделаны два DAG:
1. preprocess_data — загружает исходные данные, преобразовывает с помощью тренировочного пайплайна
2. train_model — переобучает модель на подготовленных данных

### Серввис предсказаний

Сервис прогнозирует, какие дополнительные продукты клиент банка может приобрести в следующем месяце на основе исторических данных. Он использует модель CatBoost и пайплайн обработки данных, упакованный в приложение FastAPI, развернутое с помощью Docker.

#### Запуск

Для запуска сервиса необходимо подговить aдиректорию ```artifacts/``` со следующими файлами:
- catboost_model.bin: обученная модель CatBoost.
- pipeline.pkl: Папйлайн препроцессинга данных.

Далее необходимо последовательно выполнить скрипты:
```
# Для запуска через Dockerfile используем
docker build -t bank-prediction-service .

docker run -d -p 8000:8000 --name bank-prediction-service bank-prediction-service
```

#### Использование

Запрос для проверки работспособности сервиса:
```
curl http://127.0.0.1:8000/health
```

Пример запроса к сервису предсказаний:
```
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "fecha_dato": "2016-05-28",
    "ncodpers": 123456,
    "ind_empleado": "A",
    "pais_residencia": "ES",
    "sexo": "H",
    "age": "35",
    "fecha_alta": "2012-08-10",
    "ind_nuevo": 0,
    "antiguedad": "45",
    "indrel": 1.0,
    "ult_fec_cli_1t": null,
    "indrel_1mes": "1.0",
    "tiprel_1mes": "A",
    "indresi": "S",
    "indext": "N",
    "conyuemp": null,
    "canal_entrada": "KHE",
    "indfall": "N",
    "tipodom": 1.0,
    "cod_prov": 28.0,
    "nomprov": "MADRID",
    "ind_actividad_cliente": 1.0,
    "renta": 87218.1,
    "segmento": "02 - PARTICULARES"
  }'
```
Пример ответа:
```
{"predicted_products":["Текущие счета"]}
```

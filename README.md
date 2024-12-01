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

### Загрузка данных
Скачиваем данные ```https://disk.yandex.com/d/Io0siOESo2RAaA``` в  директорию ```data```. 

### Запуск ноутбука с исследованием данных
```
jupyter lab EDA.ipynb
```

Выбранная на этой стадии модель обучена на исторических данных клиентов банка за 2015 год и способна предсказывать для данного клиента выбранные им услуги.

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

### Серввис предсказаний

Сервис прогнозирует, какие дополнительные продукты клиент банка может приобрести в следующем месяце на основе исторических данных. Он использует модель CatBoost и пайплайн обраььотки данныз, упакованный в приложение FastAPI, развернутое с помощью Docker.

#### Запуск

Для запуска сервиса необходимо подговить aдиректорию ```artifacts/``` со следующими файлами:
- catboost_model.bin: обученная модель CatBoost.
- pipeline.pkl: Папйлайн препроцессинга данных.

Далее необходимо последовательно выполнить скрипты:
```
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
-d '[
  {
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
  },
  {
    "fecha_dato": "2016-05-28",
    "ncodpers": 789012,
    "ind_empleado": "B",
    "pais_residencia": "ES",
    "sexo": "V",
    "age": "29",
    "fecha_alta": "2015-03-15",
    "ind_nuevo": 1,
    "antiguedad": "15",
    "indrel": 1.0,
    "ult_fec_cli_1t": null,
    "indrel_1mes": "1.0",
    "tiprel_1mes": "I",
    "indresi": "S",
    "indext": "N",
    "conyuemp": null,
    "canal_entrada": "KAT",
    "indfall": "N",
    "tipodom": 1.0,
    "cod_prov": 50.0,
    "nomprov": "ZARAGOZA",
    "ind_actividad_cliente": 1.0,
    "renta": 54000.0,
    "segmento": "01 - TOP"
  }
]'
```

### Airflow

Для обучение модели по расписанию на исторических данных развернем Airflow
```
docker compose up --build
```

После чего интерфейс Airflow будет доступен по адресу ```http://127.0.0.1:8080```.

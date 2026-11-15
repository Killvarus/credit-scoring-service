# Credit scoring service — MLOps вокруг скоринговой модели

Минимальный прод для модели из [credit-scoring](https://github.com/Killvarus/credit-scoring): HTTP API, Docker,
переобучение по расписанию в Airflow с проверкой данных и сравнением с текущей моделью, мониторинг дрейфа (PSI)
и метрик в Prometheus, Telegram-бот, манифесты Kubernetes и пакетный скоринг на Spark.

> Как всё устроено и почему именно так — в [GUIDE.md](GUIDE.md).

```
             ┌──────────── Airflow (еженедельно) ─────────────┐
 OpenML ───▶ │ extract → validate → train → evaluate → decide │──▶ artifacts/ (model.txt + metadata.json)
             └────────────────────────────────────────────────┘          │  champion / challenger
                                                                          ▼
 Telegram ──▶ bot ──HTTP──▶  FastAPI  /predict /predict/batch /drift /metrics ──▶ Prometheus (+ алерты)
                                    ▲
 Spark (ночной пакет) ──────────────┘ те же app.features и та же модель
```

## Что внутри

| компонент | где | что делает |
|---|---|---|
| API | [app/main.py](app/main.py) | `POST /predict`, `/predict/batch` (до 1000 заявок): PD, балл, решение, три главные причины риска (TreeSHAP); `GET /health`, `/ready`, `/model`, `/drift`, `/metrics` |
| Валидация входа | [app/schemas.py](app/schemas.py) | pydantic: диапазоны полей, лишние поля запрещены → HTTP 422 до модели |
| Признаки | [app/features.py](app/features.py) | один модуль для обучения, API и Spark — нет расхождения признаков между обучением и продом |
| Пайплайн обучения | [training/pipeline.py](training/pipeline.py) | extract → validate (схема, диапазоны, доля дефолтов) → split → train (early stopping) → evaluate (порог по прибыли) → compare_with_production → save; MLflow по желанию |
| Оркестрация | [airflow/dags/credit_scoring_retrain.py](airflow/dags/credit_scoring_retrain.py) | еженедельный DAG с ветвлением promote / skip |
| Мониторинг | [app/monitoring.py](app/monitoring.py), [monitoring/](monitoring/) | Prometheus-метрики (запросы, латентность, решения, распределение PD), PSI признаков к обучающей выборке, правила алертов |
| Бот | [bot/bot.py](bot/bot.py) | aiogram 3: `/score 50000 30 0 20000 2000` → решение и причины |
| Контейнеры | [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml) | multi-stage (api, bot), non-root, healthcheck; профили compose: monitoring, bot, tracking (MLflow), airflow |
| Kubernetes | [k8s/](k8s/) | Deployment с readiness/liveness, rolling update без простоя, HPA по CPU, ConfigMap |
| Пакетный скоринг | [batch/spark_score.py](batch/spark_score.py) | PySpark `mapInPandas` с тем же кодом признаков и моделью |
| CI | [.github/workflows/ci.yml](.github/workflows/ci.yml) | ruff + 23 теста; сборка обоих образов и смоук-тест контейнера; kubeconform для манифестов; Spark-джоба на Java 17 |

Модель: LightGBM, ROC-AUC 0.784 / Gini 0.567 на отложенных 20%, порог PD 0.146 выбран по условной прибыли
(+1 за хорошего клиента, −5 за дефолт) на валидации. Артефакт — нативный текстовый формат LightGBM (0,5 МБ), он лежит
в репозитории вместо реестра моделей, чтобы образ собирался без внешних хранилищ.

## Быстрый старт

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000          # документация: http://localhost:8000/docs
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @tests/data/example.json
```

Ответ:

```json
{"pd": 0.430, "score": 495, "decision": "reject", "threshold": 0.146, "model_version": "20261115-144612",
 "reasons": [{"feature": "delay_recent2", "contribution": 0.519},
             {"feature": "months_since_delay", "contribution": 0.286},
             {"feature": "delay_max", "contribution": 0.193}]}
```

В Docker:

```bash
docker compose up --build                          # API на :8000
docker compose --profile monitoring up --build     # + Prometheus на :9090
BOT_TOKEN=<токен от @BotFather> docker compose --profile bot up --build
```

Переобучение:

```bash
python -m training.train          # скачает данные, обучит; выкатит, только если ROC-AUC не хуже текущей больше чем на 0.005
```

## Что проверено, а что нет

- Локально: 23 теста (API, валидация, дрейф, пайплайн на синтетике, совместимость артефакта с кодом, бот), ruff,
  живой запуск uvicorn и запрос к `/predict`.
- В CI на GitHub Actions: сборка образов, запуск контейнера и запросы к нему, валидация манифестов, Spark-джоба.
- Профили compose `airflow` и `tracking`, а также реальная работа бота с Telegram не запускались — для этого нужны
  Docker и токен бота. DAG проверяется только на компиляцию, его шаги — функции пайплайна, покрытые тестами.

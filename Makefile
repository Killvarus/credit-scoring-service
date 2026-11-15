.PHONY: install train test lint run docker compose k8s

install:          ## зависимости для разработки
	pip install -r requirements-dev.txt

train:            ## переобучить модель (OpenML → artifacts/, если не хуже текущей)
	python -m training.train

test:
	pytest -q

lint:
	ruff check .

run:              ## API на http://localhost:8000/docs
	uvicorn app.main:app --reload --port 8000

docker:
	docker build --target api -t credit-scoring-service:latest .

compose:
	docker compose --profile monitoring up --build

k8s:              ## в кластер (kind/minikube): сначала kind load docker-image credit-scoring-service:latest
	kubectl apply -f k8s/

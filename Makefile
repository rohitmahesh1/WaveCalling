SHELL := /bin/bash

.PHONY: samples models up down rebuild clean-samples clean-models

samples:
	bash ./scripts/fetch_samples.sh

models:
	bash ./scripts/fetch_models.sh

up: samples models
	docker compose up --build

down:
	docker compose down

rebuild:
	docker compose down
	docker compose build --no-cache
	docker compose up

clean-samples:
	rm -rf samples/*
	@mkdir -p samples && touch samples/.gitkeep

clean-models:
	rm -rf export/*
	@mkdir -p export && touch export/.gitkeep

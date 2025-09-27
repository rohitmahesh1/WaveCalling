SHELL := /bin/bash

.PHONY: samples up down rebuild clean-samples

samples:
	bash ./scripts/fetch_samples.sh

up: samples
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

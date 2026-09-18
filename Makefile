.PHONY: install test init ingest backtest clean

install:
	pip install -e ".[dev]"

test:
	PYTHONPATH=src pytest -q

init:
	PYTHONPATH=src python -m bet.cli init

ingest:
	PYTHONPATH=src python -m bet.cli ingest --source all --seasons 2015-2026

backtest:
	PYTHONPATH=src python -m bet.cli backtest --from 2018-08-01

clean:
	rm -rf data/db/*.duckdb data/db/*.wal .pytest_cache

.PHONY: test fidelity unit backtest sweep lint clean install

PY := python3

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

fidelity:
	$(PY) -m pytest -q -m fidelity

unit:
	$(PY) -m pytest -q -m unit

backtest:
	$(PY) backtest.py --data backtester/tests/fixtures/synthetic_day.csv --trader example_trader.py --out out/backtest

sweep:
	$(PY) sweep.py --data backtester/tests/fixtures/synthetic_day.csv --trader example_trader.py --sweep-config backtester/tests/fixtures/sweep_config.yaml --out out/sweep

lint:
	$(PY) -m py_compile backtester/*.py backtest.py sweep.py example_trader.py

clean:
	rm -rf out .pytest_cache __pycache__ backtester/__pycache__ backtester/tests/__pycache__ backtester/tests/unit/__pycache__ backtester/tests/fidelity/__pycache__

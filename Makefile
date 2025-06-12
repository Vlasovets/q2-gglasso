.PHONY: all lint dev install clean distclean test

all: install

lint:
	q2lint
	flake8

dev: all
	pip install -e ".[dev]"

install:
	pip install -e .

test: all
	python -m pytest

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -exec rm -f {} +
	find . -name "*.pyo" -exec rm -f {} +
	find . -name "*.pyd" -exec rm -f {} +
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".coverage" -exec rm -f {} +

distclean: clean
